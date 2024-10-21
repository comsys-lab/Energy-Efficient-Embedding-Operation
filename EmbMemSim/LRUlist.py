import numpy as np

class Node:
    def __init__(self, addr=None):
        self.addr = addr  # memory address
        self.next = None  # Next node pointer (if there is no next node, it becomes None)

class LRUlist:
    def __init__(self, cache_way):
        self.head = Node(None)
        self.cache_way = cache_way # Length of the linked list == cache_way
        
        # Init the linked list with null (number of nodes==cache_way).
        current = self.head
        for _ in range(0, cache_way):
            current.next = Node(None)
            current = current.next

    def get_cache_way(self):
        return self.cache_way

    def insert_node(self, value):
        new_node = Node(value)
        if not self.head:  # if list is empty
            self.head = new_node
            return

        # if the list already has a head node.
        new_node.next = self.head
        self.head = new_node

        current = self.head
        prev = None
        count = 0
        while current and count < self.cache_way:
            prev = current
            current = current.next
            count += 1
        
        # Tail 노드 삭제 (prev는 tail 앞의 노드)
        if prev and prev.next:
            prev.next = None

    # 리스트에서 특정 값을 찾고, 찾은 노드를 head로 옮기기
    def search_list(self, value):
        current = self.head
        prev = None

        # 노드를 찾는 과정
        while current:
            if current.addr == value:
                if prev:  # 노드가 head가 아니라면
                    prev.next = current.next  # 노드를 리스트에서 제거
                    current.next = self.head  # 찾은 노드를 head로 이동
                    self.head = current
                return True  # 값을 찾았으면 True 반환
            prev = current
            current = current.next

        return False  # 값을 찾지 못했으면 False 반환

    # 연결 리스트 출력 (디버깅 용)
    def print_list(self):
        current = self.head
        while current:
            print(current.addr, end=" -> ")
            current = current.next
        print("End")