from __future__ import annotations
from typing_extensions import Optional, Union

class Node:
    def __init__(self, val: float) -> None:
        self.val = val
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None

class BST:
    def __init__(self) -> None:
        self.root: Optional[Node] = None

    def show_path(self, ins_path: list[Node]) -> str:
        return ' -> '.join([str(i.val) for i in ins_path])

    def insert(self,
               value: float,
               show_ins_path: bool = False) -> Union[int, list[Node]]:

        if show_ins_path:
            ins_path: list[Node] = []

        if self.root is None:
            self.root = Node(val)
            if show_ins_path:
                return [self.root]
            return 0

        node: Node = root

        while True:
            if show_ins_path:
                ins_path.append(node)
            if value < node.val:
                cand = node.left
                if cand is None:
                    new = Node(value)
                    node.left = new
                    break
            elif value > node.val:
                cand = node.right
                if cand is None:
                    new = Node(value)
                    node.right = new
                    break
            else: # Found same value
                raise ValueError(f'Value {value} already exists')
            node = cand

        if show_ins_path:
            ins_path.append(new)
            return ins_path
        return 0

    def search(self,
               value: float) -> bool:

        node: Node = self.root
        while node:
            # print(value, node.val)
            if value < node.val: node = node.left
            elif value > node.val: node = node.right
            else: return True
        return False

    def delete(self,
               value: float) -> int:
        parent = None
        node: Node = self.root
        mov = -1

        while node:
            # print(value, node.val)
            if value < node.val:
                parent = node
                node = node.left
                mov = 0
            elif value > node.val:
                parent = node
                node = node.right
                mov = 1
            else:
                break

        # didn't find node to be deleted
        if node is None:
            return 1

        # found the node to be deleted
        if node.left is None and node.right is None:
            if parent is None:
                self.root = None

            # leaf node deletion
            elif mov: parent.right = None
            else: parent.left = None

        elif node.left is None or node.right is None:
            # one-child node deletion
            if node.left is None:
                new_child = node.right
            else:
                new_child = node.left
            if parent is None: # node was root
                # node.val = new_child.val
                # node.right = new_child.right
                # node.left = new_child.left
                self.root = new_child
            elif mov == 1: # node was on parent's right
                parent.right = new_child
            else: # node was on parent's left
                parent.left = new_child

        else:
            # two-child node deletion
            # get next inorder

            next_in_order_parent = node
            next_in_order = node.right

            successor_is_deeper = next_in_order.left is not None # false
            while next_in_order.left:
                next_in_order_parent = next_in_order
                next_in_order = next_in_order.left

            node.val = next_in_order.val

            if successor_is_deeper:
                next_in_order_parent.left = next_in_order.right
            else:
                next_in_order_parent.right = next_in_order.right

        return 0

    def inorder_rec(self, node, delim=', '):
        if node is None: return
        self.inorder_rec(node.left, delim)
        print(node.val, end=f'{delim}')
        self.inorder_rec(node.right, delim)

    def preorder_rec(self, node, delim=', '):
        if node is None: return
        print(node.val, end=f'{delim}')
        self.preorder_rec(node.left, delim)
        self.preorder_rec(node.right, delim)

    def postorder_rec(self, node, delim=', '):
        if node is None: return
        self.postorder_rec(node.left, delim)
        self.postorder_rec(node.right, delim)
        print(node.val, end=f'{delim}')

    def inorder(self, root = None, delim=', '):
        self.inorder_rec(root if root is not None else self.root, delim=delim)
        print()

    def preorder(self, root = None, delim=', '):
        self.preorder_rec(root if root is not None else self.root, delim=delim)
        print()

    def postorder(self, root = None, delim=', '):
        self.postorder_rec(root if root is not None else self.root, delim=delim)
        print()


# CASE B:
# inorder successor is DEEP in the right subtree
# AND successor itself has a RIGHT CHILD

bst = BST()
bst.insert(50)

bst.insert(30)
bst.insert(80)

bst.insert(20)
bst.insert(40)

bst.insert(70)
bst.insert(90)

bst.insert(60)
bst.insert(75)

bst.insert(55)
bst.insert(65)

bst.insert(57)

# Tree:
#
#             50
#           /    \
#         30      80
#        / \     /  \
#      20  40   70   90
#              / \
#             60 75
#            / \
#           55 65
#             \
#             57
#
#
# inorder successor of 50 = 55
#
# BUT:
#
# 55 has a right child: 57
#
# So after copying/replacing 50 with 55,
# deleting the original 55 must reconnect:
#
#      60
#     /
#    55
#      \
#      57
#
# into:
#
#      60
#     /
#    57

bst.delete(50)

bst.inorder()
bst.preorder()
bst.postorder()
