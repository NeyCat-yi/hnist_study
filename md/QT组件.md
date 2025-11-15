没问题！下面是一份**Qt Designer（Qt Widgets Designer）组件速查文档**（面向 Qt 6 / PySide6 / C++），按类别归纳常用控件、容器、布局、以及 Designer 的关键面板与工作流。每一类都给出典型类名，便于在代码里对接。文末附上官方文档入口。

# 1) 输入与选择类（Input & Choice）

- 按钮：`QPushButton`（普通按钮）、`QToolButton`（工具栏小按钮）、`QCheckBox`（复选）、`QRadioButton`（单选）、`QCommandLinkButton`。([Qt 文档](https://doc.qt.io/qt-6/widget-classes.html?utm_source=chatgpt.com))
- 文本输入：`QLineEdit`（单行）、`QTextEdit` / `QPlainTextEdit`（多行）。([Qt 文档](https://doc.qt.io/qt-6/widget-classes.html?utm_source=chatgpt.com))
- 数值与日期：`QSpinBox` / `QDoubleSpinBox`、`QSlider`、`QDial`、`QProgressBar`、`QDateEdit` / `QTimeEdit` / `QDateTimeEdit`。([Qt 文档](https://doc.qt.io/qt-6/widget-classes.html?utm_source=chatgpt.com))
- 下拉与选择：`QComboBox`、`QFontComboBox`、`QColorDialog`（在代码中调用的对话框）。([Qt 文档](https://doc.qt.io/qt-6/widget-classes.html?utm_source=chatgpt.com))

# 2) 显示与分组（Display & Grouping）

- 标签与展示：`QLabel`、`QLCDNumber`、`QFrame`（分隔线/框）。([Qt 文档](https://doc.qt.io/qt-6/widget-classes.html?utm_source=chatgpt.com))
- 分组与页签：`QGroupBox`（分组框）、`QTabWidget`（多页签）、`QToolBox`（抽屉式页）、`QStackedWidget`（堆叠页，配合按钮/列表切换）。([Qt 文档](https://doc.qt.io/qt-6/widget-classes.html?utm_source=chatgpt.com))
- 停靠与分割：`QSplitter`（可拖拽分割）、`QDockWidget`（主窗口停靠部件，仅用于 `QMainWindow`）。([Qt 文档](https://doc.qt.io/qt-6/widget-classes.html?utm_source=chatgpt.com))

# 3) 列表/表格/树（Item Views）

- 视图类：`QListView`、`QTableView`、`QTreeView`（MVC 通用），以及便捷部件 `QListWidget`、`QTableWidget`、`QTreeWidget`（自带模型，适合小型应用）。([Qt 文档](https://doc.qt.io/qt-6/widget-classes.html?utm_source=chatgpt.com))

# 4) 容器与窗口（Containers & Windows）

- 顶层窗口：`QMainWindow`（主窗体，支持菜单栏/工具栏/状态栏/停靠区）、`QDialog`（对话框）、`QWidget`（通用容器）。([Qt 文档](https://doc.qt.io/qt-6/qtwidgets-index.html?utm_source=chatgpt.com))
- 主窗口专属：`QMenuBar`、`QToolBar`、`QStatusBar`；在 Designer 里选“Main Window”即可自动带这些区域。([Qt 文档](https://doc.qt.io/qt-6/qtdesigner-index.html?utm_source=chatgpt.com))

# 5) 布局（Layouts）

- 线性布局：`QHBoxLayout`、`QVBoxLayout`；网格：`QGridLayout`；表单：`QFormLayout`。在 Designer 中先选中父容器，再点工具栏“水平/垂直/网格/表单布局”。嵌套布局是常态。([Qt 文档](https://doc.qt.io/qt-6/designer-layouts.html?utm_source=chatgpt.com))
- 伸缩与间隔：`QSpacerItem`（水平/垂直弹性空白），配合 Size Policy（最小/首选/扩展）精细控制自适应。([Qt 文档](https://doc.qt.io/qt-6/designer-layouts.html?utm_source=chatgpt.com))

# 6) Designer 的核心面板/模式（工作区）

- **Widget Box**：所有可拖拽的控件库。
- **Property Editor**：所选控件的属性（对象名、文本、对齐、Size Policy、样式表等）。
- **Object Inspector**：对象树，便于选中父子层级。
- **Action Editor**：集中管理 `QAction`（菜单项/工具按钮复用）。
- **编辑模式**：
  - Widget Editing（摆放控件）
  - Signals/Slots（连线信号槽）
  - Buddy（为无障碍/快速导航设置伙伴，如标签→输入框）
  - Tab Order（调整 Tab 焦点顺序） ([Qt 文档](https://doc.qt.io/qt-6/qtdesigner-manual.html?utm_source=chatgpt.com))

# 7) 资源与样式

- 资源：使用 `.qrc` 管理图片/图标，Designer 里在属性中即可选资源。([Qt 文档](https://doc.qt.io/qt-6/qtdesigner-manual.html?utm_source=chatgpt.com))
- 样式表（Qt Style Sheets）：在 Property Editor 的 `styleSheet` 属性编写 CSS 风格规则。([Qt 文档](https://doc.qt.io/qt-6/qtdesigner-manual.html?utm_source=chatgpt.com))

# 8) 自定义与推广（Promote）

- **Promoted Widgets**：先在 Designer 放一个基类（如 `QWidget` / `QLabel`），再把它“Promote”为你自定义的子类（如 `MyPlotWidget`），无需做 Designer 插件也能在运行时替换。([Qt 文档](https://doc.qt.io/qt-6/ja/designer-using-custom-widgets.html?utm_source=chatgpt.com))
- 若需要在 Designer 面板中“原生”出现你的控件，可制作 **Designer 插件**。([Qt 文档](https://doc.qt.io/qt-6/qtdesigner-manual.html?utm_source=chatgpt.com))

# 9) 在代码中加载 .ui

- **C++（Qt 6 Widgets）**：`QUiLoader`（或将 `.ui` 编译为 `ui_xxx.h` 并在类里 `setupUi(this)`）。官方手册有完整流程。([Qt 文档](https://doc.qt.io/qt-6/qtdesigner-manual.html?utm_source=chatgpt.com))
- **PySide6**：常用两种方式：
  1. 直接使用 `QUiLoader` / `QFile` 动态载入；
  2. 用 `pyside6-uic` 先把 `.ui` 转成 `.py` 再 `setupUi`。模块与类索引见 Qt for Python 文档。([Qt 文档](https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/index.html?utm_source=chatgpt.com))

# 10) 快速实践建议

- **先布局，后细调**：给顶层容器设置主布局；嵌套小布局组织控件；再调 Size Policy 与最小/最大尺寸。([Qt 文档](https://doc.qt.io/qt-6/designer-layouts.html?utm_source=chatgpt.com))
- **尽量用视图+模型**：数据量稍大时优先 `QTableView + QAbstractTableModel`，避免 `QTableWidget` 性能瓶颈。([Qt 文档](https://doc.qt.io/qt-6/widget-classes.html?utm_source=chatgpt.com))
- **动作统一管理**：菜单、工具栏按钮都用同一个 `QAction`，减少重复。([Qt 文档](https://doc.qt.io/qt-6/qtdesigner-manual.html?utm_source=chatgpt.com))
- **信号槽在 Designer 连或在代码连都行**：复杂逻辑建议在代码里统一连接，便于版本控制。([Qt 文档](https://doc.qt.io/qt-6/qtdesigner-manual.html?utm_source=chatgpt.com))

------

## 官方文档入口（建议收藏）

- Qt Widgets Designer 手册（Qt 6）：表单编辑模式、布局、容器、资源、样式表、在 C++/Python 中使用 .ui 等。([Qt 文档](https://doc.qt.io/qt-6/qtdesigner-manual.html?utm_source=chatgpt.com))
- Qt Widgets（类与模块索引、控件全列表）：查具体控件属性/信号槽/用法。([Qt 文档](https://doc.qt.io/qt-6/qtwidgets-index.html?utm_source=chatgpt.com))
- Qt for Python（PySide6）Widgets 索引：Python 侧类名与 API。([Qt 文档](https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/index.html?utm_source=chatgpt.com))
- 布局使用指南（Designer 视角）：常见布局与嵌套技巧。([Qt 文档](https://doc.qt.io/qt-6/designer-layouts.html?utm_source=chatgpt.com))

如果你告诉我打算做的界面草图或列出要用的组件，我可以直接给你一份 **Qt Designer 操作步骤 + 代码骨架（PySide6/C++）**。