# Maintainer's Guide

This document provides instructions for Google employees to maintain the codebase, specifically managing the copybara sync process. This document was adapted from instructions originally shared by Judah.

## Setup Copybara

Follow the `go/copybara-setup` instructions to setup copybara on your Google machine.

This involves setting a bash alias for the `copybara` CLI.


## Update Config

Ensure the "copy.bara.sky" file in the root directory of the repository is up to date. This file exists in the Google codebase only.

## Dry Run

### GitHub to Google

Perform a dry run sync:

```
copybara third_party/py/smart_buildings/copy.bara.sky  --init-history --dry-run --force
```

This creates a CL on the Google side, pulling in all the chagnes from GitHub to Google.

Remember to merge this CL on the Google side before proceeding!

### Google to GitHub

Invoke the “git_to_third_party” workflow in the .sky file.

This will overwrite the GitHub repository, so make sure to merge the changes from GitHub to Google first!
