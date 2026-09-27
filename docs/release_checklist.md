# Publication checklist

The source package and original model are prepared for review. A public GitHub
remote has not been created by this release operation.

- Confirm repository owner/name and add the real URL to citation metadata.
- Confirm inherited code attribution and the redistribution terms for model
  weights and media; preserve `NOTICE` and asset provenance.
- Review `VALIDATION.md`. Run the documented learned-policy/ROS smoke check in
  the actual Humble/Stonefish environment and record dependency versions and
  external scenario revisions.
- Enable the included CI workflow and inspect its actual results before adding
  a passing badge or expanding supported-version claims.
- Keep campaign bags and large training runs outside ordinary Git history.
  Link a versioned data archive when available.
- Add a version tag after reviewing the repository and test results. Add a
  paper DOI only once a verified publication record exists.

The real project demo is included. No repeat of the research campaigns is
required to publish the code; the remaining runtime check concerns this cleaned
wrapper, not the validity of the historical campaign records.
