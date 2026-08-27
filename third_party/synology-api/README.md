# Synology API vendored-wheel source

Production installs the checked-in, hash-locked wheel at
`requirements/vendor/synology_api-0.9.1-py3-none-any.whl`. This directory makes
that wheel independently reproducible after retirement of the AI4S GitHub fork.

The source is defined by:

1. upstream `N4S4/synology-api` tag `v0.9.1` at the exact commit and tree in
   `source.lock.json`;
2. the single binary-capable patch in `patches/`;
3. the exact patched tree, wheel digest, `SOURCE_DATE_EPOCH`, and build command
   recorded in `requirements/synology-api-wheel.json`.

The MIT license is copied byte-for-byte from the patched source. Use
`scripts/repository/materialize_third_party_patch.py` to verify the patch and
materialize the source without changing the upstream checkout.
