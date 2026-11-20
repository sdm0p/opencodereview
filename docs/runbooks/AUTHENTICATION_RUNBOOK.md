# GitHub authentication runbook

The CLI already supports GitHub OAuth Device Flow through
`GITHUB_OAUTH_CLIENT_ID`. Web authentication is the final production gate and
must be enabled only after the core review and posting checks pass.

## Required deployment values

- `GITHUB_OAUTH_CLIENT_ID`: GitHub OAuth App client ID.
- `GITHUB_OAUTH_CLIENT_SECRET`: OAuth App secret stored as a platform secret.
- `OCR_REQUIRE_AUTH=true`: require authenticated API requests in production.
- `OCR_ALLOWED_ORIGINS`: the exact Vercel frontend origin.

## GitHub permissions

Use the least privilege needed to read pull requests and create pull-request
or issue comments. Validate the granted identity before creating a review and
store the user ID with every job. A user may read, approve, or post only jobs
owned by that identity.

## Final acceptance checks

1. Complete OAuth login and callback with a test GitHub account.
2. Create a review from the authenticated frontend.
3. Confirm the job is owned by that account in Supabase.
4. Approve the review and verify the GitHub posting receipt and comments.
5. Confirm a second account cannot read or approve the job.
6. Revoke the token and confirm subsequent requests fail safely.

Never commit OAuth secrets or personal access tokens. Configure them only in
the deployment secret manager.
