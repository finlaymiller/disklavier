import os
import omegaconf
from PySide6 import QtWidgets
from utils import console

blocked_params = [
    "n_beats_per_segment",
    "tag",
    "lead_bar",
    "tick_1",
    "tick_2",
    "sample_rate",
    "verbose",
    "midi_port",
    "record",
    "channels",
    "system",
    "user",
    "remote_host",
    "port",
    "port_name",
    "remote_dir",
    "startup_delay",
]

key_order = [
    "bpm",
    "n_transitions",
    "initialization",
    "player_tracking",
    "seeker",
    "augmentation",
    "scheduler",
    "player",
    "recorder",
    "audio",
]


class ParameterEditorWidget(QtWidgets.QWidget):
    def __init__(self, params, parent=None):
        super().__init__(parent)

        self.param_options = {
            "initialization": ["kickstart", "random", "recording"],
            "seeker.mode": [
                "best",
                "random",
                "sequential",
                "graph",
                "probabilities",
            ],
            "seeker.metric": [
                "pitch-histogram",
                "specdiff",
                "clamp",
                # "clf-4note",
                # "clf-speed",
                # "clf-tpose",
            ],
            "seeker.match": [
                "current",
                "next",
                "prev",
                "next 2",
                "prev 2",
            ],
            "recorder.pedal_type": [
                "regular",
                "inverted",
            ],
            "player_tracking": [
                None,
                "threshold",
                "threshold (transposed)",
                "weighted average",
            ],
        }

        self.params = params
        self.param_widgets = {}

        # cleaner ui during demo
        if self.params.seeker.system == "demo":
            blocked_params.append("kickstart_path")
            blocked_params.append("graph_steps")
            blocked_params.append("graph_track_revisit_interval")
            blocked_params.append("metric")
            blocked_params.append("probability_transition_lookback")
            blocked_params.append("max_velocity")
            blocked_params.append("velocity_proportion")
            blocked_params.append("min_expected_velocity")
            blocked_params.append("max_expected_velocity")
            blocked_params.append("min_adjustment")
            blocked_params.append("max_adjustment")
            blocked_params.append("cc_number")
            blocked_params.append("min")
            blocked_params.append("max")
            blocked_params.append("middle_c_note_number")

        self.init_ui()

    def init_ui(self):
        """
        initialize the user interface.
        """
        main_layout = QtWidgets.QVBoxLayout(self)

        # larger text
        self.setStyleSheet(
            """
            QLabel {
                font-size: 14px;
            }
            QLineEdit {
                font-size: 14px;
            }
            QComboBox {
                font-size: 14px;
            }
            QCheckBox {
                font-size: 14px;
            }
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 5px;
            }
        """
        )

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)

        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout(container)
        container_layout.setSpacing(10)

        # header
        header_frame = QtWidgets.QFrame()
        header_layout = QtWidgets.QHBoxLayout(header_frame)
        header_layout.setContentsMargins(0, 0, 0, 0)

        param_label = QtWidgets.QLabel("Parameter")
        param_label.setFixedWidth(300)
        value_label = QtWidgets.QLabel("Value")
        value_label.setFixedWidth(400)

        header_layout.addWidget(param_label)
        header_layout.addWidget(value_label)

        container_layout.addWidget(header_frame)

        # display parameters
        self.display_params(container_layout)

        scroll_area.setWidget(container)
        main_layout.addWidget(scroll_area)

    def display_params(self, layout):
        """
        display parameters in the gui, supporting up to two levels of nesting for groups.
        """
        if not self.params:
            empty_label = QtWidgets.QLabel("No parameters loaded.")
            layout.addWidget(empty_label)
            return

        self._add_widgets_for_level(layout, self.params, [])

    def _add_widgets_for_level(self, parent_layout, current_level_params, path_parts):
        """
        recursively add widgets for parameters.

        this helper populates `parent_layout` with parameter rows or nested group boxes
        based on `current_level_params`. `path_parts` tracks the nesting.
        groups are created for dictionary parameters up to two levels deep.

        Parameters
        ----------
        parent_layout : QtWidgets.QLayout
            the layout to add widgets to.
        current_level_params : dict or omegaconf.dictconfig.DictConfig
            the parameters for the current nesting level.
        path_parts : list[str]
            a list of keys representing the path to `current_level_params`.

        Returns
        -------
        bool
            true if any widgets were added to `parent_layout`, false otherwise.
        """
        keys_to_display = list(current_level_params.keys())

        if not path_parts:  # only apply key_order sort at the top level
            keys_to_display.sort(
                key=lambda x: (
                    key_order.index(x) if x in key_order else len(key_order),
                    x,
                )
            )
        else:
            keys_to_display.sort()  # sorts alphabetically for sub-levels

        items_added_on_this_level = False

        for key in keys_to_display:
            current_full_key_parts = path_parts + [key]
            full_key_str = ".".join(current_full_key_parts)

            # skip if the key itself or its full path is in blocked_params
            if key in blocked_params or full_key_str in blocked_params:
                continue

            value = current_level_params[key]

            if isinstance(value, (dict, omegaconf.dictconfig.DictConfig)):
                if (
                    len(path_parts) < 2
                ):  # create a group box if we are at level 0 or level 1 of path_parts
                    formatted_section_title = key.replace("_", " ").title()
                    section_group = QtWidgets.QGroupBox(formatted_section_title)
                    section_layout = QtWidgets.QVBoxLayout(section_group)

                    children_added = self._add_widgets_for_level(
                        section_layout, value, current_full_key_parts
                    )

                    if children_added:  # only add the group if it has content
                        parent_layout.addWidget(section_group)
                        items_added_on_this_level = True
                else:
                    # we are already 2 levels deep in groups. add parameters from this dict directly.
                    children_added = self._add_widgets_for_level(
                        parent_layout, value, current_full_key_parts
                    )
                    if children_added:
                        items_added_on_this_level = True
            else:  # assumed to be a primitive, list, or other type that add_param_row can handle
                self.add_param_row(parent_layout, full_key_str, value)
                items_added_on_this_level = True

        return items_added_on_this_level

    def add_param_row(self, layout, key, value) -> None:
        """
        add a parameter row to the gui.

        Parameters
        ----------
        layout : QLayout
            layout to add the row to.
        key : str
            parameter key.
        value : any
            parameter value.
        """
        frame = QtWidgets.QFrame()
        row_layout = QtWidgets.QHBoxLayout(frame)
        row_layout.setContentsMargins(0, 2, 0, 2)

        # format display key
        display_key = key.split(".")[-1]
        display_key = display_key.replace("_", " ").title()

        label = QtWidgets.QLabel(display_key)
        label.setFixedWidth(300)
        row_layout.addWidget(label)

        # use appropriate widget based on parameter type and options
        if key == "seeker.probabilities_dist":
            # display list as comma-separated string
            str_value = ", ".join(map(str, value))
            widget = QtWidgets.QLineEdit(str_value)
            widget.setFixedWidth(400)
            row_layout.addWidget(widget)
            self.param_widgets[key] = widget
        elif key in self.param_options:
            widget = QtWidgets.QComboBox()
            widget.addItems(self.param_options[key])
            widget.setCurrentText(str(value))
            widget.setFixedWidth(400)
            row_layout.addWidget(widget)
            self.param_widgets[key] = widget
        elif isinstance(value, bool):
            value_container = QtWidgets.QWidget()
            value_container.setFixedWidth(400)
            inner_layout = QtWidgets.QHBoxLayout(value_container)
            inner_layout.setContentsMargins(0, 0, 0, 0)
            inner_layout.addSpacing(4)
            widget = QtWidgets.QCheckBox()
            widget.setChecked(value)
            inner_layout.addWidget(widget)
            inner_layout.addStretch(1)
            row_layout.addWidget(value_container)
            self.param_widgets[key] = widget
        elif isinstance(value, (int, float)):
            widget = QtWidgets.QLineEdit(str(value))
            widget.setFixedWidth(400)
            row_layout.addWidget(widget)
            self.param_widgets[key] = widget
        else:
            widget = QtWidgets.QLineEdit(str(value))
            widget.setFixedWidth(400)
            row_layout.addWidget(widget)
            self.param_widgets[key] = widget

        layout.addWidget(frame)

    def get_updated_params(self):
        """
        get updated parameter values from the widgets.

        returns
        -------
        dict
            updated parameters.
        """
        for key, widget in self.param_widgets.items():
            value = None

            if isinstance(widget, QtWidgets.QComboBox):
                value = widget.currentText()
            elif isinstance(widget, QtWidgets.QCheckBox):
                value = widget.isChecked()
            elif isinstance(widget, QtWidgets.QLineEdit):
                text_value = widget.text()
                original_value = self.get_param_value(key)

                if key == "seeker.probabilities_dist":
                    try:
                        parsed_values = [
                            float(x.strip()) for x in text_value.split(",")
                        ]
                        if len(parsed_values) == 6:
                            value = parsed_values
                        else:
                            console.log(
                                f"[red]Error: seeker.probabilities_dist requires 6 comma-separated numbers. "
                                f"Got: {text_value}. Reverting to original.[/red]"
                            )
                            value = original_value  # revert
                    except ValueError:
                        console.log(
                            f"[red]Error: Invalid input for seeker.probabilities_dist. "
                            f"Expected comma-separated numbers. Got: {text_value}. Reverting to original.[/red]"
                        )
                        value = original_value  # revert
                elif isinstance(original_value, int):
                    try:
                        value = int(text_value)
                    except ValueError:
                        value = original_value
                elif isinstance(original_value, float):
                    try:
                        value = float(text_value)
                    except ValueError:
                        value = original_value
                else:
                    value = text_value

            if value is not None:
                self.set_param_value(key, value)

        return self.params

    def get_param_value(self, key_path):
        """
        get parameter value from the config using a dot-separated key path.

        Parameters
        ----------
        key_path : str
            parameter key path (e.g., "section.subsection.key").

        Returns
        -------
        any
            parameter value, or none if the path is invalid.
        """
        keys = key_path.split(".")
        current_value = self.params
        try:
            for k_segment in keys:
                current_value = current_value[k_segment]
            return current_value
        except (KeyError, TypeError, omegaconf.errors.ConfigKeyError):
            # handle cases where path is invalid or value is not subscriptable
            console.log(
                f"[red]Error: Could not retrieve value for key path: {key_path}[/red]"
            )
            return None

    def set_param_value(self, key_path, new_value):
        """
        set parameter value in the config using a dot-separated key path.

        Parameters
        ----------
        key_path : str
            parameter key path (e.g., "section.subsection.key").
        new_value : any
            parameter value to set.
        """
        keys = key_path.split(".")
        obj = self.params
        try:
            for k_segment in keys[:-1]:  # navigate to the parent dictionary
                obj = obj[k_segment]

            # attempt to convert type if original was int/float and new value is string representable as such
            original_type = type(obj[keys[-1]]) if keys[-1] in obj else None

            if original_type is int and isinstance(new_value, str):
                try:
                    new_value = int(new_value)
                except ValueError:
                    pass  # keep as string if conversion fails, omegaconf might handle or error later
            elif original_type is float and isinstance(new_value, str):
                try:
                    new_value = float(new_value)
                except ValueError:
                    pass  # keep as string

            obj[keys[-1]] = new_value  # set the value on the final key
        except (KeyError, TypeError, omegaconf.errors.ConfigKeyError):
            # handle cases where path is invalid or obj is not subscriptable
            console.log(
                f"[red]Error: Could not set value for key path: {key_path}[/red]"
            )
