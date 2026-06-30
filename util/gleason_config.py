from util.adversarial_candidates import (
    adversarial_variable_choices,
    get_adversarial_candidate,
)


def parse_adversarial_specs(value):
    if value is None or value.strip() == "":
        return {}

    specs = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        name, num_classes = item.split(":")
        name = name.strip()
        if "." in name:
            raise ValueError("Adversarial head names cannot contain '.'")
        specs[name] = int(num_classes)
    if len(specs) > 1:
        raise ValueError("Use one adversarial variable per run")
    return specs


ADVERSARIAL_VARIABLE_CHOICES = adversarial_variable_choices()


def default_binned_column(variable):
    candidate = get_adversarial_candidate(variable)
    if candidate:
        return candidate["target_column"]
    return variable if variable.endswith("_bin") else f"{variable}_bin"


def default_observed_column(variable):
    candidate = get_adversarial_candidate(variable)
    if candidate:
        return candidate["observed_column"]
    return variable if variable.endswith("_observed") else f"{variable}_observed"


def resolve_adversarial_config(args):
    legacy_specs = parse_adversarial_specs(args.adversarial_specs)
    variable = args.adversarial_variable

    if variable is None:
        specs = legacy_specs
        columns = {name: name for name in specs}
        observed_columns = {name: default_observed_column(name) for name in specs}
    else:
        if legacy_specs:
            raise ValueError(
                "Use either --adversarial_variable or --adversarial_specs, not both"
            )
        candidate = get_adversarial_candidate(variable)
        candidate_num_classes = candidate.get("adversarial_num_classes")
        if (
            candidate_num_classes is not None
            and args.adversarial_num_classes is not None
            and int(args.adversarial_num_classes) != int(candidate_num_classes)
        ):
            raise ValueError(
                f"--adversarial_num_classes={args.adversarial_num_classes} conflicts "
                f"with configured value {candidate_num_classes} for "
                f"--adversarial_variable {variable}. Update "
                "util/adversarial_candidates.py or remove the CLI override."
            )
        num_classes = candidate_num_classes or args.adversarial_num_classes
        if num_classes is None:
            raise ValueError(
                "--adversarial_num_classes is required when --adversarial_variable is set"
            )
        args.adversarial_num_classes = int(num_classes)
        specs = {variable: int(num_classes)}
        columns = {variable: args.adversarial_column or default_binned_column(variable)}
        observed_columns = {
            variable: getattr(args, "adversarial_observed_column", None)
            or default_observed_column(variable)
        }

    if specs:
        args.adversarial_specs_resolved = specs
        args.adversarial_columns = columns
        args.adversarial_observed_columns = observed_columns
        args.adversarial_definition = ", ".join(
            f"{name}:{columns[name]}:{observed_columns[name]}:{num_classes}"
            for name, num_classes in specs.items()
        )
    else:
        args.adversarial_specs_resolved = {}
        args.adversarial_columns = {}
        args.adversarial_observed_columns = {}
        args.adversarial_definition = "none"
    return args.adversarial_specs_resolved
