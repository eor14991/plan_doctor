"""
Infrastructure: Firebase Admin SDK Initialisation
==================================================
This module is the single point of contact with the Firebase Admin SDK.
All other modules receive the Firestore client as a constructor argument
and remain unaware of how it was initialised.

Credential resolution order:
    1. If FIREBASE_SERVICE_ACCOUNT_FILE is set, credentials are loaded from
       that JSON file path. This is the recommended approach for local
       development and containerised deployments that mount secret files.
    2. If FIREBASE_SERVICE_ACCOUNT is set, credentials are parsed from the
       JSON string inline. This supports environments where secrets are
       injected as environment variable strings (e.g. CI/CD pipelines).

Security note:
    The credential file is read and parsed here in the infrastructure layer.
    No path or credential data propagates into the core domain or adapters.
    Swapping the credential source requires only a change to this module and
    the Settings class, with no impact on any other layer.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import firebase_admin
from firebase_admin import credentials, firestore_async

logger = logging.getLogger(__name__)

_firestore_client = None


def initialize_firebase(
    service_account_json: Optional[str] = None,
    service_account_file: Optional[str] = None,
) -> None:
    """
    Initialise the Firebase Admin SDK exactly once per process.

    If the SDK has already been initialised, this function logs a message
    and returns without performing any action.

    Args:
        service_account_json: The full service account JSON serialised as a
                              string. Used when credentials are injected via
                              the FIREBASE_SERVICE_ACCOUNT environment variable.
        service_account_file: Absolute or relative path to a service account
                              JSON key file. Used when credentials are stored
                              as a file on disk.

    Raises:
        ValueError: If neither argument is provided.
        FileNotFoundError: If service_account_file is provided but does not exist.
        json.JSONDecodeError: If the JSON string or file content is malformed.
    """
    try:
        firebase_admin.get_app()
        logger.info("Firebase Admin SDK already initialised.")
        return
    except ValueError:
        pass  # Not yet initialised — proceed.

    if service_account_file:
        resolved_path = os.path.abspath(service_account_file)
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Firebase service account file not found: {resolved_path}")
        cred = credentials.Certificate(resolved_path)
        logger.info("Firebase initialised from file.", extra={"path": resolved_path})
    elif service_account_json:
        service_account_info = json.loads(service_account_json)
        cred = credentials.Certificate(service_account_info)
        logger.info("Firebase initialised from environment variable JSON.")
    else:
        raise ValueError(
            "Firebase credentials not provided. Set either "
            "FIREBASE_SERVICE_ACCOUNT_FILE or FIREBASE_SERVICE_ACCOUNT."
        )

    firebase_admin.initialize_app(cred)


def get_firestore_client():
    """
    Return the async Firestore client, creating it on the first call.

    initialize_firebase() must be called before this function is invoked.

    Returns:
        An async Firestore client (google.cloud.firestore_v1.AsyncClient).
    """
    global _firestore_client
    if _firestore_client is None:
        _firestore_client = firestore_async.client()
    return _firestore_client


def verify_firebase_token(id_token: str) -> dict:
    """
    Verify a Firebase ID token and return the decoded claims.

    Args:
        id_token: The raw JWT string from the Authorization header.

    Returns:
        A dictionary of decoded token claims including 'uid' and 'email'.

    Raises:
        firebase_admin.auth.ExpiredIdTokenError: If the token has expired.
        firebase_admin.auth.InvalidIdTokenError: If the token is malformed
            or the signature is invalid.
    """
    from firebase_admin import auth

    return auth.verify_id_token(id_token)
