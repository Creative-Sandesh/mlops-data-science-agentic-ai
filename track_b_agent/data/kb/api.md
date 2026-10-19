# Nimbus Cloud Platform - API Limits

Rate limits are enforced per user: Starter allows 60 requests per minute, Pro allows 600 requests per minute and Enterprise allows 6000 requests per minute.

Requests above the limit receive HTTP 429 with a Retry-After header.

API keys expire and must be rotated every 90 days. Each user can hold at most 5 active API keys. The maximum request payload is 10 MB.
