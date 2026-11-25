# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | ✅ Yes     |

## Reporting a Vulnerability

We take the security of ScamShield AI seriously. If you discover a security vulnerability, please report it responsibly.

### How to Report

1. **Do NOT** create a public GitHub issue
2. Email security findings to: security@scamshield.ai
3. Include detailed reproduction steps
4. Allow 48 hours for initial response

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Impact assessment
- Suggested fix (if applicable)
- Your contact information

## Security Best Practices

### For Developers

1. Use environment variables for all secrets
2. Never commit `.env` files
3. Run `make lint` before pushing code
4. Keep dependencies updated
5. Use HTTPS in production
6. Enable rate limiting
7. Validate all user inputs

### For Production Deployment

1. Change default SECRET_KEY
2. Use PostgreSQL instead of SQLite
3. Enable TLS/SSL for database connections
4. Configure proper CORS policies
5. Set up monitoring and alerting
6. Regular security audits
7. Backup database regularly

## Security Features

- ✅ JWT Authentication with token refresh
- ✅ Password hashing with bcrypt
- ✅ Rate limiting per endpoint
- ✅ Input validation with Pydantic
- ✅ SQL injection prevention via ORM
- ✅ CORS protection
- ✅ CSRF protection
- ✅ Audit logging
- ✅ Data encryption at rest
- ✅ Secure headers middleware

## Known Issues

No known security vulnerabilities at this time.

## References

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [Python Security](https://docs.python.org/3/library/security.html)