#class for linked list

class Node:
    def __init__(data):
        self.data = data
        self.next = 0

Class LL:
    def __init__(self, head):
        self.head = head

    def add_node(self, node):
        if head not None:
            node.next = head
        else:
            head = node
        head = node
        print("Node added")

    def show_LL(self):
        curr = head
        if curr is not None:
            while curr:
                print(curr.data)
                curr = curr.next
        else:
            print("No data")
