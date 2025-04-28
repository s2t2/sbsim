# Maintainer's Guide

This document provides instructions for Google employees to maintain the codebase, specifically managing the copybara sync process. This document was adapted from instructions originally shared by Judah.

## Setup Copybara

Follow the `go/copybara-setup` instructions to setup copybara on your Google machine.

This involves setting a bash alias for the `copybara` CLI.

If successful, these commands should resolve without error:

```sh
copybara version

copybara help
```


## Update Config

Ensure the "copy.bara.sky" file in the root directory of the repository is up to date. This file exists in the Google codebase only.

Create and/or activate a workspace:

```sh
gcert

# create new workspace and navigate to google3 dir:
g4d -f mjr_copybara_20250428

# as necessary, pull updated config file from a CL that hasn't been merged yet:
# g4 patch -c <CL_number>
# g4 sync
# g4 change

#cd third_party/py/smart_buildings/
```



## Dry Run

### GitHub to Google

Perform a dry run sync:

```sh
# copybara third_party/py/smart_buildings/copy.bara.sky  --init-history --dry-run --force
copybara copy.bara.sky  --init-history --dry-run

```

This creates a CL on the Google side, pulling in all the chagnes from GitHub to Google.

Remember to merge this CL on the Google side before proceeding!

### Google to GitHub

Invoke the “git_to_third_party” workflow in the .sky file.

This will overwrite the GitHub repository, so make sure to merge the changes from GitHub to Google first!
