# Releasing

Two one-time setups are needed before the first release. Both require the owner's
accounts.

## 1. PyPI trusted publishing (one time)

Trusted publishing lets GitHub Actions upload to PyPI without any token being stored in
this repository. On <https://pypi.org/manage/account/publishing/>, add a pending publisher.

| Field | Value |
|---|---|
| PyPI project name | `thermospike` |
| Owner | `tlcdv` |
| Repository name | `thermospike` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

Then, in this repository's settings, create an environment called `pypi`. No secret is
needed in it.

## 2. Zenodo (one time)

A DOI makes the library citable. On <https://zenodo.org/account/settings/github/>, sign in
with GitHub and switch on `tlcdv/thermospike`. Zenodo then mints a DOI for every release
from that point on.

After the first release, add the DOI badge to `README.md` and the DOI to `CITATION.cff`.

## Making a release

1. Check that `main` is green on the tests workflow.
2. Bump the version in `pyproject.toml`, `src/thermospike/__init__.py` and
   `CITATION.cff`. All three must agree.
3. Run `python -m pytest -q` and `python examples/compare_cost.py`. If the numbers in the
   README changed, update the table from the new output.
4. Commit, then create a GitHub release with the tag `vX.Y.Z`. The tag and the package
   version must match. `publish.yml` checks this and fails the release if they differ.
5. Publishing to PyPI and minting the DOI then happen automatically.
