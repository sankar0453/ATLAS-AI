"""
Test cases for ATLAS-AI .env configuration.

Run (no extra installs needed):
    python test/test.py
or, if pytest is installed:
    pytest test/test.py -v

Offline checks validate presence, format and consistency of every key.
Live connectivity checks (AWS, S3, Bedrock, Pinecone, Postgres) are skipped
unless RUN_LIVE_TESTS=1 is set in the shell environment.
"""

import os
import re
import unittest
from pathlib import Path
from urllib.parse import urlparse

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"

ENV = dotenv_values(ENV_PATH) if ENV_PATH.exists() else {}

EXPECTED_KEYS = {
    # AWS
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION",
    "S3_BUCKET",
    # Bedrock
    "BEDROCK_CHAT_MODEL_ID",
    "BEDROCK_EMBED_MODEL_ID",
    "BEDROCK_EMBED_DIMENSIONS",
    "BEDROCK_MAX_TOKENS",
    "BEDROCK_EFFORT",
    # Pinecone
    "PINECONE_API_KEY",
    "PINECONE_INDEX_HOST",
    "PINECONE_CLOUD",
    "PINECONE_REGION",
    "PINECONE_READY_TIMEOUT",
    # Postgres
    "DATABASE_URL",
    # RUIS observability
    "RUIS_API_KEY",
    "RUIS_SERVICE_NAME",
    "RUIS_CAPTURE_CONTENT",
    "RUIS_DISABLED",
}

BOOL_VALUES = {"true", "false", "1", "0", "yes", "no"}
AWS_REGION_RE = re.compile(r"^[a-z]{2}(-gov)?-[a-z]+-\d$")
LIVE = os.getenv("RUN_LIVE_TESTS") == "1"


def get(key):
    return (ENV.get(key) or "").strip()


def as_bool(value):
    return value.strip().lower() in {"true", "1", "yes"}


# ---------------------------------------------------------------------------
# File-level checks
# ---------------------------------------------------------------------------
class TestEnvFiles(unittest.TestCase):
    def test_env_file_exists(self):
        self.assertTrue(ENV_PATH.exists(), ".env file is missing")

    def test_env_has_all_expected_keys(self):
        missing = EXPECTED_KEYS - ENV.keys()
        self.assertFalse(missing, f".env missing keys: {sorted(missing)}")

    def test_no_empty_values_in_env(self):
        empty = sorted(k for k in EXPECTED_KEYS if not get(k))
        self.assertFalse(empty, f".env has empty values for: {empty}")

    def test_env_is_gitignored(self):
        gitignore = (ROOT / ".gitignore").read_text()
        self.assertRegex(gitignore, r"(?m)^\.env\s*$", ".env must be in .gitignore")


# ---------------------------------------------------------------------------
# AWS
# ---------------------------------------------------------------------------
class TestAwsConfig(unittest.TestCase):
    def test_access_key_id_format(self):
        self.assertRegex(get("AWS_ACCESS_KEY_ID"), r"^(AKIA|ASIA)[A-Z0-9]{16}$")

    def test_secret_access_key_format(self):
        self.assertRegex(get("AWS_SECRET_ACCESS_KEY"), r"^[A-Za-z0-9/+=]{40}$")

    def test_region_format(self):
        self.assertRegex(get("AWS_REGION"), AWS_REGION_RE)

    def test_s3_bucket_name_rules(self):
        bucket = get("S3_BUCKET")
        self.assertTrue(3 <= len(bucket) <= 63, "bucket name must be 3-63 chars")
        self.assertRegex(bucket, r"^[a-z0-9][a-z0-9.-]*[a-z0-9]$")
        self.assertNotIn("..", bucket)
        self.assertNotRegex(bucket, r"^\d+\.\d+\.\d+\.\d+$", "bucket cannot be an IP")


# ---------------------------------------------------------------------------
# Bedrock
# ---------------------------------------------------------------------------
class TestBedrockConfig(unittest.TestCase):
    MODEL_ID_RE = r"^[a-z0-9-]+\.[a-zA-Z0-9.\-]+(:\d+)?$"

    def test_chat_model_id_format(self):
        self.assertRegex(get("BEDROCK_CHAT_MODEL_ID"), self.MODEL_ID_RE)

    def test_embed_model_id_format(self):
        self.assertRegex(get("BEDROCK_EMBED_MODEL_ID"), self.MODEL_ID_RE)

    def test_embed_model_is_embedding_model(self):
        self.assertIn("embed", get("BEDROCK_EMBED_MODEL_ID").lower())

    def test_embed_dimensions_is_supported(self):
        dims = int(get("BEDROCK_EMBED_DIMENSIONS"))
        if "titan-embed-text-v2" in get("BEDROCK_EMBED_MODEL_ID"):
            self.assertIn(dims, {256, 512, 1024}, "Titan v2 supports 256/512/1024")
        else:
            self.assertGreater(dims, 0)

    def test_max_tokens_is_positive_int(self):
        tokens = int(get("BEDROCK_MAX_TOKENS"))
        self.assertGreater(tokens, 0)
        self.assertLessEqual(tokens, 200_000)

    def test_effort_is_valid(self):
        self.assertIn(get("BEDROCK_EFFORT").lower(), {"low", "medium", "high"})


# ---------------------------------------------------------------------------
# Pinecone
# ---------------------------------------------------------------------------
class TestPineconeConfig(unittest.TestCase):
    def test_api_key_present(self):
        self.assertGreaterEqual(len(get("PINECONE_API_KEY")), 20)

    def test_index_host_format(self):
        host = get("PINECONE_INDEX_HOST")
        hostname = urlparse(host).hostname if "://" in host else host
        self.assertTrue(hostname, "PINECONE_INDEX_HOST has no hostname")
        self.assertTrue(hostname.endswith(".pinecone.io"), f"unexpected host: {hostname}")

    def test_cloud_is_valid(self):
        self.assertIn(get("PINECONE_CLOUD").lower(), {"aws", "gcp", "azure"})

    def test_region_format(self):
        if get("PINECONE_CLOUD").lower() == "aws":
            self.assertRegex(get("PINECONE_REGION"), AWS_REGION_RE)
        else:
            self.assertTrue(get("PINECONE_REGION"))

    def test_ready_timeout_is_positive_int(self):
        self.assertGreater(int(get("PINECONE_READY_TIMEOUT")), 0)


# ---------------------------------------------------------------------------
# Postgres
# ---------------------------------------------------------------------------
class TestDatabaseConfig(unittest.TestCase):
    def setUp(self):
        self.url = urlparse(get("DATABASE_URL"))

    def test_scheme_is_postgres(self):
        self.assertIn(self.url.scheme, {"postgres", "postgresql", "postgresql+psycopg"})

    def test_has_host(self):
        self.assertTrue(self.url.hostname, "DATABASE_URL has no host")

    def test_has_credentials(self):
        self.assertTrue(self.url.username, "DATABASE_URL has no username")
        self.assertTrue(self.url.password, "DATABASE_URL has no password")

    def test_has_database_name(self):
        self.assertTrue(self.url.path.strip("/"), "DATABASE_URL has no database name")

    def test_port_is_valid_if_set(self):
        if self.url.port is not None:
            self.assertTrue(1 <= self.url.port <= 65535)


# ---------------------------------------------------------------------------
# RUIS observability
# ---------------------------------------------------------------------------
class TestRuisConfig(unittest.TestCase):
    def test_api_key_present_when_enabled(self):
        if not as_bool(get("RUIS_DISABLED")):
            self.assertTrue(get("RUIS_API_KEY"), "RUIS enabled but RUIS_API_KEY empty")

    def test_service_name_format(self):
        self.assertRegex(get("RUIS_SERVICE_NAME"), r"^[A-Za-z0-9_.-]+$")

    def test_capture_content_is_bool(self):
        self.assertIn(get("RUIS_CAPTURE_CONTENT").lower(), BOOL_VALUES)

    def test_disabled_is_bool(self):
        self.assertIn(get("RUIS_DISABLED").lower(), BOOL_VALUES)


# ---------------------------------------------------------------------------
# Live connectivity (opt-in: RUN_LIVE_TESTS=1)
# ---------------------------------------------------------------------------
@unittest.skipUnless(LIVE, "set RUN_LIVE_TESTS=1 to run live connectivity tests")
class TestLiveConnectivity(unittest.TestCase):
    def _session(self):
        import boto3
        return boto3.Session(
            aws_access_key_id=get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=get("AWS_SECRET_ACCESS_KEY"),
            region_name=get("AWS_REGION"),
        )

    def test_aws_credentials_valid(self):
        identity = self._session().client("sts").get_caller_identity()
        self.assertIn("Account", identity)

    def test_s3_bucket_accessible(self):
        self._session().client("s3").head_bucket(Bucket=get("S3_BUCKET"))

    def test_bedrock_models_available(self):
        bedrock = self._session().client("bedrock")
        for key in ("BEDROCK_CHAT_MODEL_ID", "BEDROCK_EMBED_MODEL_ID"):
            with self.subTest(key=key):
                model = bedrock.get_foundation_model(modelIdentifier=get(key))
                self.assertEqual(model["modelDetails"]["modelId"], get(key))

    def test_embedding_dimension_matches(self):
        import json
        runtime = self._session().client("bedrock-runtime")
        resp = runtime.invoke_model(
            modelId=get("BEDROCK_EMBED_MODEL_ID"),
            body=json.dumps({
                "inputText": "ping",
                "dimensions": int(get("BEDROCK_EMBED_DIMENSIONS")),
            }),
        )
        embedding = json.loads(resp["body"].read())["embedding"]
        self.assertEqual(len(embedding), int(get("BEDROCK_EMBED_DIMENSIONS")))

    def test_pinecone_index_reachable_and_dimension_matches(self):
        from pinecone import Pinecone
        index = Pinecone(api_key=get("PINECONE_API_KEY")).Index(host=get("PINECONE_INDEX_HOST"))
        stats = index.describe_index_stats()
        self.assertEqual(stats["dimension"], int(get("BEDROCK_EMBED_DIMENSIONS")))

    def test_postgres_connects(self):
        import psycopg
        with psycopg.connect(get("DATABASE_URL"), connect_timeout=10) as conn:
            self.assertEqual(conn.execute("SELECT 1").fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
