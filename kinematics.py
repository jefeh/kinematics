import re
import ast
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout,
                               QHeaderView, QLabel, QMainWindow, QSlider,
                               QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget, QFileDialog, QSpinBox, QMessageBox)

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

class Expression:
    """ Represents a mathematical expression composed of floats, operators, and variables. Variables in the expression must match keys in the provided variables dict. """
    def __init__(self, expr: str):
        self.expr_str = expr.replace('$', '')
        self._ast = ast.parse(self.expr_str, mode='eval')

    def eval(self, variables: Dict[str, float]) -> float:
        """
        Evaluate the expression given a mapping of variable names to float values.
        """
        return Expression._eval_ast(self._ast.body, variables)

    @staticmethod
    def _eval_ast(node: ast.AST, variables: Dict[str, float]) -> float:
        if isinstance(node, ast.BinOp):
            left = Expression._eval_ast(node.left, variables)
            right = Expression._eval_ast(node.right, variables)
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if isinstance(node.op, ast.Div): return left / right
            if isinstance(node.op, ast.Pow): return left ** right
            raise ValueError(f"Unsupported binary operator: {node.op}")
        elif isinstance(node, ast.UnaryOp):
            val = Expression._eval_ast(node.operand, variables)
            if isinstance(node.op, ast.UAdd): return +val
            if isinstance(node.op, ast.USub): return -val
            raise ValueError(f"Unsupported unary operator: {node.op}")
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        elif isinstance(node, ast.Name):
            if node.id in variables: return variables[node.id]
            raise ValueError(f"Unknown variable '{node.id}' in expression")
        else:
            raise ValueError(f"Unsupported AST node: {node}")

@dataclass
class Variable:
    name: str
    description: str
    default: float

@dataclass
class Joint:
    id: int
    type: str
    min_expr: Expression
    max_expr: Expression
    offset_expr: Expression
    encoder: int
    factor: float
    prev: int

class Kinematics:
    """ Computes coordinate frames for a chain of joints given variable and encoder values. """
    def __init__(self, joints: List[Joint], variables: Dict[str, Variable]):
        self.joints = sorted(joints, key=lambda j: j.id)
        self.variables = variables

    def compute_frames(self, var_values: Dict[str, float], encoder_values: Dict[int, int]) -> Dict[int, np.ndarray]:
        # update variable values
        for name, val in var_values.items():
            if name in self.variables:
                self.variables[name].value = val
            else:
                raise KeyError(f"Unknown variable: {name}")
        # prepare var eval map
        eval_map = {name: var.value for name, var in self.variables.items()}
        frames: Dict[int, np.ndarray] = {}
        # world origin frame
        frames[0] = np.eye(4)
        for joint in self.joints:
            prev_frame = frames[joint.prev]
            # evaluate expressions
            min_val = joint.min_expr.eval(eval_map)
            offset_val = joint.offset_expr.eval(eval_map)
            enc_val = encoder_values.get(joint.encoder, 0)
            numeric = min_val + offset_val + enc_val * joint.factor
            # build transformation
            T = np.eye(4)
            if joint.type == 'LINEAL':
                T[2, 3] = numeric
            else:
                theta = np.deg2rad(numeric)
                if joint.type == 'PITCH':
                    c, s = np.cos(theta), np.sin(theta)
                    T = np.array([[ c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])
                elif joint.type == 'ROLL':
                    c, s = np.cos(theta), np.sin(theta)
                    T = np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])
                elif joint.type == 'YAW':
                    c, s = np.cos(theta), np.sin(theta)
                    T = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            frames[joint.id] = prev_frame @ T
        return frames

                
class KinematicParser:
    SECTION_RE = re.compile(r"^\s*\[(?P<section>[^\]]*)\]\s*$")
    VAR_RE = re.compile(r"^\s*(?P<name>[A-Za-z_]\w*)\s*=\s*'(?P<desc>[^']*)'\s*,\s*(?P<default>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s*$")

    def __init__(self):
        self.variables: Dict[str, Variable] = {}
        self.joints: List[Joint] = []

    def parse(self, filename: str) -> Kinematics:
        current_section: Optional[str] = None
        with open(filename, 'r') as f:
            for raw in f:
                line = raw.split('#', 1)[0].strip()
                if not line: continue
                sec_match = self.SECTION_RE.match(line)
                if sec_match:
                    current_section = sec_match.group('section')
                    continue
                if current_section == 'Vars': self._parse_var_line(line)
                elif current_section == 'Joints': self._parse_joint_line(line)
        return Kinematics(self.joints, self.variables)

    def _parse_var_line(self, line: str):
        m = self.VAR_RE.match(line)
        if not m: raise ValueError(f"Invalid Vars line: {line}")
        name, desc, default = m.group('name'), m.group('desc'), float(m.group('default'))
        self.variables[name] = Variable(name, desc, default)

    def _parse_joint_line(self, line: str):
        parts = [p.strip() for p in line.split('|')]
        # pad to length 8
        while len(parts) < 8: 
                parts.append('')
        id_joint = int(parts[0])
        node_type = parts[1]
        # defaults
        default_max = sys.float_info.max
        # parse expressions or defaults
        min_expr = Expression(parts[2]) if parts[2] else Expression('0')
        max_expr = Expression(parts[3]) if parts[3] else Expression(str(default_max))
        offset_expr = Expression(parts[4]) if parts[4] else Expression('0')
        encoder = int(parts[5]) if parts[5] else 0
        factor = float(parts[6]) if parts[6] else 1.0
        prev = int(parts[7])

        if any(j.id == id_joint for j in self.joints):
            raise ValueError(f"Duplicate joint id: {id_joint}")
        if node_type not in ('LINEAL', 'YAW', 'PITCH', 'ROLL'):
            raise ValueError(f"Unsupported joint type: {node_type}")
        self.joints.append(Joint(id_joint, node_type, min_expr, max_expr, offset_expr, encoder, factor, prev))


class ApplicationWindow(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setWindowTitle('Kinematics')
        
        # Initialize members
        self.kinematics = None
        self.variables: Dict[str, float] = {}
        self.encoders: Dict[int, int] = {}

        # Main menu bar
        self._create_menu()

        # Figure (Left)
        self.fig = Figure(figsize=(5, 5))
        self.canvas = FigureCanvas(self.fig)

        # Controls
        self._create_controls()

    def _create_menu(self):
        menu = self.menuBar()
        file_menu = menu.addMenu('File')
        load = QAction('Load', self, triggered=self._load_file)
        exit = QAction('Exit', self, triggered=qApp.quit)
        file_menu.addAction(load); file_menu.addAction(exit)
        about_menu = menu.addMenu('&About')
        about = QAction('About Qt', self, triggered=qApp.aboutQt)
        about_menu.addAction(about)

    def _create_controls(self):
        self._main = QWidget()
        self.setCentralWidget(self._main)
        self.llayout = QVBoxLayout()
        self.rlayout = QVBoxLayout()
        self.llayout.setContentsMargins(1,1,1,1)
        self.rlayout.setContentsMargins(1,1,1,1)
        if self.kinematics:
            self.llayout.addWidget(self.canvas, 1)
            self.encoders = {j.encoder:0 for j in self.kinematics.joints if j.encoder>0}
            for enc in self.encoders:
                spin_layout = QHBoxLayout()
                spin_layout.addWidget(QLabel(f'Encoder {enc}'), 1)
                spin = QSpinBox()
                spin.setRange(-32732,32732)
                spin.valueChanged.connect(lambda val, e=enc: self._update_encoder(e,val))
                spin_layout.addWidget(spin, 3)
                self.llayout.addLayout(spin_layout)
            self.variables = {v.name:v.default for v in self.kinematics.variables.values()}
            for var in self.kinematics.variables.values():
                vlayout = QVBoxLayout()
                vlayout.addWidget(QLabel(var.description))
                spin = QSpinBox()
                spin.setRange(-32732,32732)
                spin.setValue(var.default)
                spin.valueChanged.connect(lambda val, n=var.name: self._update_variable(n,val))
                vlayout.addWidget(spin)
                self.rlayout.addLayout(vlayout)
        layout = QHBoxLayout(self._main)
        layout.addLayout(self.llayout, 70)
        layout.addLayout(self.rlayout, 30)

    def _load_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Kinematics', '', 'Kinematics Files (*.mdef)')
        if fname:
            try:
                parser = KinematicParser()
                self.kinematics = parser.parse(fname)
            except Exception:
                QMessageBox.critical(self, 'Error', 'Could not read file.', QMessageBox.Ok)
        self._create_controls(); self._refresh()

    def _update_variable(self, name: str, value: float):
        if name in self.variables: self.variables[name] = value
        self._refresh()

    def _update_encoder(self, encoder_num: int, value: int):
        if encoder_num in self.encoders: self.encoders[encoder_num] = value
        self._refresh()

    def _refresh(self):
        if not self.kinematics: return
        results = self.kinematics.compute_frames(self.variables, self.encoders)
        self._render_results(results)

    def _render_results(self, results: Dict[int, np.ndarray]):
        # Clear figure and create 3D axes
        self.fig.clear()
        ax = self.fig.add_subplot(111, projection='3d')
        # Map joint id to its predecessor for drawing links
        id_prev = {j.id: j.prev for j in self.kinematics.joints}
        # TCPs
        tcps = set(id_prev.keys()) - set(id_prev.values())
        # Draw each frame
        for jid, frame in results.items():
            origin = frame[:3, 3]
            if jid in tcps:
                # Determine axis length
                dist = np.linalg.norm(origin)
                length = dist * 0.2 if dist > 0 else 0.1
                # X axis (red)
                ax.quiver(*origin, *frame[:3,0], length=length, normalize=True, color='r')
                # Y axis (green)
                ax.quiver(*origin, *frame[:3,1], length=length, normalize=True, color='g')
                # Z axis (blue)
                ax.quiver(*origin, *frame[:3,2], length=length, normalize=True, color='b')
            # Draw link from previous
            if jid != 0:
                prev_id = id_prev.get(jid, 0)
                p0 = results[prev_id][:3,3]
                p1 = origin
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color='k')
        # Labels and aspect
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #ax.set_box_aspect((1,1,1))
        ax.set_aspect('equal', 'box')
        #self.fig.tight_layout()
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = ApplicationWindow()
    w.setFixedSize(1280, 720)
    w.show()
    app.exec()
