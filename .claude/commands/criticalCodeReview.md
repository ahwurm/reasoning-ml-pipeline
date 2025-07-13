# Critical Code Review Protocol

## Security Review
- Check for SQL injection vulnerabilities
- Validate input sanitization
- Review authentication/authorization
- Check for sensitive data exposure
- Validate CORS configuration
- Review error message information leakage

## Performance Review
- Analyze algorithmic complexity
- Check for memory leaks
- Review database query efficiency
- Validate caching strategies
- Check for unnecessary re-renders (React)
- Review bundle size impact

## Code Quality Review
- Verify proper error handling
- Check for code duplication
- Review naming conventions
- Validate type safety
- Check for proper logging
- Review test coverage

## Architecture Review
- Validate separation of concerns
- Check for proper abstraction layers
- Review API design consistency
- Validate data flow patterns
- Check for proper dependency injection

## Review Checklist
- [ ] All functions have proper error handling
- [ ] Input validation is comprehensive
- [ ] Database queries are optimized
- [ ] Security best practices followed
- [ ] Performance benchmarks met
- [ ] Test coverage above 80%
- [ ] Documentation is up to date
- [ ] No console.log/println! statements in production
- [ ] Proper logging implemented
- [ ] Code follows style guidelines