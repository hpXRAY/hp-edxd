from PyQt6.QtCore import *
from PyQt6.QtGui import QDrag
from PyQt6.QtWidgets import *


class TreeWidget(QTreeWidget):
    customMimeType = "application/x-customTreeWidgetdata"

    def mimeTypes(self):
        mimetypes = QTreeWidget.mimeTypes(self)
        mimetypes.append(TreeWidget.customMimeType)
        return mimetypes

    def startDrag(self, supportedActions):
        drag = QDrag(self)
        mimedata = self.model().mimeData(self.selectedIndexes())

        encoded = QByteArray()
        stream = QDataStream(encoded, QIODevice.WriteOnly)
        self.encodeData(self.selectedItems(), stream)
        mimedata.setData(TreeWidget.customMimeType, encoded)

        drag.setMimeData(mimedata)
        drag.exec_(supportedActions)

    def dropEvent(self, event):
        if event.source() == self:
            event.setDropAction(Qt.MoveAction)
            QTreeWidget.dropEvent(self, event)
        elif isinstance(event.source(), QTreeWidget):
            if event.mimeData().hasFormat(TreeWidget.customMimeType):
                encoded = event.mimeData().data(TreeWidget.customMimeType)
                parent = self.itemAt(event.pos())
                items = self.decodeData(encoded, event.source())
                for it in items:
                    item = QTreeWidgetItem(parent)
                    self.fillItem(it, item)
                    self.fillItems(it, item)
                event.acceptProposedAction()

    def fillItem(self, inItem, outItem):
        for col in range(inItem.columnCount()):
            for key in range(Qt.UserRole):
                role = Qt.ItemDataRole(key)
                outItem.setData(col, role, inItem.data(col, role))

    def fillItems(self, itFrom, itTo):
        for ix in range(itFrom.childCount()):
            it = QTreeWidgetItem(itTo)
            ch = itFrom.child(ix)
            self.fillItem(ch, it)
            self.fillItems(ch, it)

    def encodeData(self, items, stream):
        stream.writeInt32(len(items))
        for item in items:
            p = item
            rows = []
            while p is not None:
                rows.append(self.indexFromItem(p).row())
                p = p.parent()
            stream.writeInt32(len(rows))
            for row in reversed(rows):
                stream.writeInt32(row)
        return stream

    def decodeData(self, encoded, tree):
        items = []
        rows = []
        stream = QDataStream(encoded, QIODevice.ReadOnly)
        while not stream.atEnd():
            nItems = stream.readInt32()
            for i in range(nItems):
                path = stream.readInt32()
                row = []
                for j in range(path):
                    row.append(stream.readInt32())
                rows.append(row)

        for row in rows:
            it = tree.topLevelItem(row[0])
            for ix in row[1:]:
                it = it.child(ix)
            items.append(it)
        return items


def create_treeWidget(name, ncolumn):
    treeWidget = TreeWidget()
    header = QTreeWidgetItem([str(i) for i in range(ncolumn)])
    treeWidget.setHeaderItem(header)
    root = QTreeWidgetItem(treeWidget, [name])
    for letter in ["A", "B", "C"]:
        item = QTreeWidgetItem(root, ["{}-{}".format(name, letter)])
        QTreeWidgetItem(item, ["{}-{}-{}".format(name, letter, i) for i in range(ncolumn)])
    treeWidget.expandAll()
    treeWidget.setSelectionMode(QAbstractItemView.MultiSelection)
    treeWidget.setDragEnabled(True)
    treeWidget.viewport().setAcceptDrops(True)
    treeWidget.setDropIndicatorShown(True)
    return treeWidget


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    app.setStyle("fusion")
    w1 = create_treeWidget("tree1", 3)
    w2 = create_treeWidget("tree2", 3)
    w1.show()
    w2.show()
    sys.exit(app.exec_())