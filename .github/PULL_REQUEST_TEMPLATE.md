# Pull Request

## Description

<!-- Provide a brief description of the changes in this PR -->

## Type of Change

<!-- Mark the relevant option with an "x" -->

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring
- [ ] Test improvements
- [ ] CI/CD improvements

## Related Issues

<!-- Link to related issues using keywords like "Fixes #123", "Closes #456", "Related to #789" -->

Fixes #

## Changes Made

<!-- Provide a detailed list of changes -->

-
-
-

## Testing

### Test Coverage

- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Test coverage maintained or improved
- [ ] Manual testing completed

### Testing Details

<!-- Describe the tests you ran and their results -->

```bash
# Commands used for testing
poetry run pytest
poetry run coverage run -m pytest
poetry run coverage report
```

**Test Results:**
```
# Paste relevant test output here
```

## Code Quality

- [ ] Code follows the project's style guidelines (flake8)
- [ ] Type hints added/updated where appropriate
- [ ] Docstrings added/updated (Google style)
- [ ] Comments added for complex logic
- [ ] No new warnings generated
- [ ] Pre-commit hooks pass

### Quality Checks

```bash
# Commands used for quality checks
poetry run flake8 imputation_showcase tests
poetry run mypy imputation_showcase/
```

## Documentation

- [ ] README.md updated (if applicable)
- [ ] CHANGELOG.md updated
- [ ] API documentation updated (if applicable)
- [ ] Examples added/updated (if applicable)
- [ ] Notebooks updated (if applicable)

## Performance Impact

<!-- If applicable, describe any performance implications -->

- [ ] No performance impact
- [ ] Performance improved
- [ ] Performance degraded (justified below)

**Details:**
<!-- Provide benchmarks or explain performance changes -->

## Breaking Changes

<!-- If this PR includes breaking changes, describe them and the migration path -->

- [ ] No breaking changes
- [ ] Breaking changes (documented below)

**Migration Guide:**
<!-- Provide instructions for users to migrate from the old API to the new one -->

## Screenshots/Visualizations

<!-- If applicable, add screenshots or visualizations showing the changes -->

## Checklist

### Before Submitting

- [ ] I have read the [CONTRIBUTING.md](../CONTRIBUTING.md) guidelines
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

### Code Quality

- [ ] Code is DRY (Don't Repeat Yourself)
- [ ] Code follows SOLID principles where applicable
- [ ] Edge cases are handled
- [ ] Error handling is appropriate
- [ ] Logging is appropriate (if applicable)

### Security

- [ ] No sensitive information (passwords, tokens, keys) included
- [ ] Input validation added where necessary
- [ ] No SQL injection vulnerabilities
- [ ] No XSS vulnerabilities
- [ ] Dependencies updated to secure versions (if applicable)

## Additional Notes

<!-- Any additional information that reviewers should know -->

## Reviewer Guidance

<!-- Help reviewers by pointing out areas that need special attention -->

**Focus Areas:**
-
-

**Questions for Reviewers:**
-
-

---

By submitting this pull request, I confirm that my contribution is made under the terms of the MIT License and I agree to follow the project's [Code of Conduct](../CODE_OF_CONDUCT.md).
