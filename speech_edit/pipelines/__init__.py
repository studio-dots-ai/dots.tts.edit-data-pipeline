"""Pipeline configuration shared by distributed and HTTP entrypoints."""

from omegaconf import OmegaConf


def load_parameters(config_path: str, overrides: dict | None = None) -> dict:
    """Load the explicitly selected pipeline defaults and merge caller overrides."""
    from speech_edit.runtime.config import load_run_config

    config = load_run_config(config_path, resolve=False)
    if config_path.startswith("service/"):
        services = config["service"]
        params = next(iter(services.values()))["params"]
        defaults = load_parameters(params["pipeline_defaults"], params["pipeline"])
    else:
        defaults = config["pipeline_config"]
    return OmegaConf.to_container(OmegaConf.merge(defaults, overrides or {}), resolve=True)


def resolve_request_parameters(request, parameters, controls, aliases=None):
    """Resolve configured controls once, preserving explicit request overrides."""
    aliases = aliases or {}
    values = request.model_dump(exclude_unset=True)
    for name in controls:
        if values.get(name) is None:
            values[name] = parameters[aliases.get(name, name)]
    return type(request).model_validate(values)
