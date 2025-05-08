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
]

key_order = [
    "bpm",
    "n_transitions",
    "seeker",
    "seed_rearrange",
    "seed_remove",
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
        }

        self.params = params
        self.param_widgets = {}

        self.init_ui()

    def init_ui(self):
        """
        initialize the user interface.
        """
        main_layout = QtWidgets.QVBoxLayout(self)

        # Scroll area setup
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)

        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout(container)
        container_layout.setSpacing(10)

        # Create header
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

        # Display parameters
        self.display_params(container_layout)

        scroll_area.setWidget(container)
        main_layout.addWidget(scroll_area)

    def display_params(self, layout):
        """
        display parameters in the gui.
        """
        if not self.params:
            empty_label = QtWidgets.QLabel("No parameters loaded.")
            layout.addWidget(empty_label)
            return

        # Display top-level parameters
        keys = list(self.params.keys())
        keys.sort(key=lambda x: key_order.index(x) if x in key_order else len(key_order))
        for key in keys:
            if (
                isinstance(self.params[key], (int, float, str, bool))
                and key not in blocked_params
            ):
                self.add_param_row(layout, key, self.params[key])
            elif isinstance(
                self.params[key],
                (dict, omegaconf.OmegaConf, omegaconf.dictconfig.DictConfig),
            ):
                # Create section header
                section_group = QtWidgets.QGroupBox(key)
                section_layout = QtWidgets.QVBoxLayout(section_group)

                # Add nested parameters
                key_added = False
                for subkey in self.params[key].keys():
                    if subkey not in blocked_params:
                        full_key = f"{key}.{subkey}"
                        self.add_param_row(
                            section_layout, full_key, self.params[key][subkey]
                        )
                        key_added = True

                if key_added:
                    layout.addWidget(section_group)

    def add_param_row(self, layout, key, value):
        """
        add a parameter row to the gui.

        parameters
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

        label = QtWidgets.QLabel(key)
        label.setFixedWidth(300)
        row_layout.addWidget(label)

        # Use appropriate widget based on parameter type and options
        if key in self.param_options:
            widget = QtWidgets.QComboBox()
            widget.addItems(self.param_options[key])
            widget.setCurrentText(str(value))
            widget.setFixedWidth(400)
            row_layout.addWidget(widget)
            self.param_widgets[key] = widget
        elif isinstance(value, bool):
            widget = QtWidgets.QCheckBox()
            widget.setChecked(value)
            row_layout.addWidget(widget)
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
                if isinstance(original_value, int):
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

    def get_param_value(self, key):
        """
        get parameter value from the config.

        parameters
        ----------
        key : str
            parameter key.

        returns
        -------
        any
            parameter value.
        """
        if "." in key:
            section, subkey = key.split(".", 1)
            return self.params[section][subkey]
        return self.params[key]

    def set_param_value(self, key, value):
        """
        set parameter value in the config.

        parameters
        ----------
        key : str
            parameter key.
        value : any
            parameter value.
        """
        if "." in key:
            section, subkey = key.split(".", 1)
            self.params[section][subkey] = value
        else:
            self.params[key] = value
