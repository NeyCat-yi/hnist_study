# “左侧导航栏 + 右侧内容区”：

------

## `QListWidget` + `QStackedWidget`

### 在 Designer 里的操作步骤

1. **新建窗口**

- 选 `Main Window` 或普通 `Widget` 都行。若是 `Main Window`，中间的中心部件记得先 `Right click → Lay out` 设主布局。

1. **搭骨架**

- 拖一个 `QWidget` 进来作为根容器（或直接用已有中央部件），设置 **水平布局（QHBoxLayout）**。
- 在布局里放两个子容器：**左侧栏** 和 **右侧内容**。
  - 左：放一个 `QListWidget`（或 `QTreeWidget`）。对象名如 `navList`。
  - 右：放一个 `QStackedWidget`。对象名如 `stackedPages`。

1. **设置尺寸与自适应**

- 选中左侧 `navList`：
  - `sizePolicy`：Horizontal=**Fixed** 或 **Preferred**，Vertical=**Expanding**；
  - `minimumWidth` 设如 160；`maximumWidth` 可空或也设 160（固定宽）。
- 右侧 `stackedPages`：Horizontal=**Expanding**，Vertical=**Expanding**。
- 整体父容器必须已经有水平布局（很重要）。

1. **准备页面**

- 双击 `stackedPages`，`Insert Page` 添加多页（Page 0、Page 1…）。
- 每一页里放对应的业务控件：表格、表单、图表等。
- 在 `navList` 里添加同样数量的条目（右键 → `Edit Items...`），例如“总览 / 数据 / 设置 …”。

1. **连信号槽（纯 Designer 方式）**

- 切到 “Signals/Slots 编辑模式”（F4）。
- 从 `navList` 拖线到 `stackedPages`，选择信号 **`currentRowChanged(int)`** → 槽 **`setCurrentIndex(int)`**。
- 这样点击左侧项会切换右侧页面。

1. **细节美化（可选）**

- `navList` 属性：`viewMode=ListMode`，`flow=TopToBottom`，`movement=Static`，`selectionBehavior=SelectItems`，`selectionMode=SingleSelection`；

- 若要图标+文字：为每个条目设置 `icon`；`iconSize` 设为 20~24。

- QSS（样式表）示例（填到 `navList` 的 `styleSheet`）：

  ```css
  QListWidget {
      background:#1f2330; color:#cbd5e1; border:none;
  }
  QListWidget::item {
      height:36px; padding-left:12px;
  }
  QListWidget::item:selected {
      background:#2b3142; color:#ffffff;
  }
  QListWidget::item:hover {
      background:#252b3a;
  }
  ```

### 运行时代码骨架

PySide6 例子（加载 .ui）

```python
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile
import sys

app = QApplication(sys.argv)

ui_file = QFile("main.ui")           # 你的 Designer 文件
ui_file.open(QFile.ReadOnly)
window = QUiLoader().load(ui_file)
ui_file.close()

nav = window.findChild(type(window).findChild.__class__, "navList")  # 简便写法不可靠，下面演示更稳妥：
# 更稳妥的写法：直接按类型查
from PySide6.QtWidgets import QListWidget, QStackedWidget
nav: QListWidget = window.findChild(QListWidget, "navList")
stack: QStackedWidget = window.findChild(QStackedWidget, "stackedPages")

# 如果前面没在 Designer 里连信号槽，可以在代码里连：
nav.currentRowChanged.connect(stack.setCurrentIndex)

window.show()
sys.exit(app.exec())
```

C++ 例子（`setupUi`）

```cpp
#include <QApplication>
#include "ui_main.h"   // 由 uic 生成
#include <QListWidget>
#include <QStackedWidget>

int main(int argc, char *argv[]) {
    QApplication a(argc, argv);
    QWidget w;
    Ui::Main ui;
    ui.setupUi(&w);

    // 若没在 Designer 连信号槽：
    QObject::connect(ui.navList, &QListWidget::currentRowChanged,
                     ui.stackedPages, &QStackedWidget::setCurrentIndex);

    w.show();
    return a.exec();
}
```

> 小技巧：如果右侧每一页是你自定义的复杂控件，可以在 Designer 里先放一个普通 `QWidget`，然后用 **Promote to…** 把它提升成你的自定义类，运行时会加载你的控件（无需写 Designer 插件）。

