# forecastforge.core

The exception hierarchy shared by every stage. Library code raises these
instead of bare `ValueError` subclasses so callers can catch a single base
type while still distinguishing invalid data from misuse.

## Errors

::: forecastforge.core.errors
