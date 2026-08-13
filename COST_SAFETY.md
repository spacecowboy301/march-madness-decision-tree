# Cost Safety

## Guarantee for downloads and clones

Downloading or cloning this repository cannot charge the repository owner.
The project has no hosted backend, API key, account credential, cloud resource,
payment integration, or scheduled GitHub Actions workflow.

The main analysis uses public NCAA data files and local computation. Merely
downloading the repository does not execute code or make network requests.

## Optional KenPom scraper

KenPom access is optional and is not used as a model input in the primary
analysis. The scraper reads `KENPOM_EMAIL` and `KENPOM_PASSWORD` from the local
environment at run time. No owner credential is present in Git, generated
notebooks, reports, or configuration.

Anyone who runs the scraper must use an account they are authorized to use and
is responsible for that account's subscription and terms. A clone cannot use
the repository owner's KenPom account.

## Repository safeguards

- Credentials, cached KenPom pages, downloaded data, and local environments
  are ignored.
- `scripts/check_cost_safety.py` rejects tracked credential files, private
  keys, and common live-token formats.
- No GitHub Actions workflow performs downloads, scraping, training, or hosted
  inference.
- The trained model runs locally and does not call a metered inference API.
