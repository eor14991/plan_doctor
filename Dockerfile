# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Builder
# ─────────────────────────────────────────────────────────────────────────────
# بنستخدم multi-stage build:
#   builder: بيحمّل كل الـ dependencies (كبيرة — torch وحده ~2GB)
#   runtime: الصورة النهائية الصغيرة اللي بتشتغل على السيرفر
#
# فايدة ده: الـ build tools (gcc, pip cache) مش بتتشال في الصورة النهائية.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# build-essential: محتاجه لـ compile بعض الـ Python packages (torch extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# ── Docker Layer Caching Trick ──────────────────────────────────────────────
# بنكوبي requirements.txt الأول بس (مش الكود).
# لو الكود اتغير بس والـ requirements لأ،
# Docker بيعيد استخدام الـ pip install layer من الـ cache.
# وده بيوفر وقت كتير في الـ rebuild.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install -r requirements.txt

# دلوقتي نكوبي الكود (بعد الـ deps علشان نستفيد من الـ cache)
COPY . .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install --no-deps .

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Runtime
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# ── Security: Non-root User ───────────────────────────────────────────────────
# لو في ثغرة في التطبيق، المهاجم مش هيبقى معاه root access على الـ host.
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

# بناخد الـ installed packages من الـ builder فقط
COPY --from=builder /install /usr/local

# الكود بس
COPY src/ ./src/

# المجلدات اللي بنحفظ فيها الـ uploads والـ Qdrant vectors
RUN mkdir -p ./storage/uploads ./storage/qdrant \
    && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

# ── Health Check ──────────────────────────────────────────────────────────────
# --start-period=180s:
#   بنديه 3 دقايق علشان:
#   - BART summarization model (~1.6GB) يتحمّل
#   - BAAI/bge-m3 embedding model (~1GB) يتحمّل
#   - Qdrant يتوصل
#   لو حطيت 60s زي الأول، Docker هيعتبر الـ container unhealthy
#   وهيعيد تشغيله وانت لسه بتحمّل الموديلز!
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/docs')" || exit 1

# ── Startup Command ───────────────────────────────────────────────────────────
# --workers 1:
#   مهم جداً! لو عملت workers > 1،
#   كل worker هيحمّل الـ BART model والـ embedding model في الـ RAM.
#   موديلين × 4 workers = 10GB+ RAM على طول!
#   الـ async framework بيتعامل مع الـ concurrent requests تلقائي
#   من غير ما تحتاج workers كتير.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
