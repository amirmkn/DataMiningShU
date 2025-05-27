import pandas as pd

class FPTreeNode:
    def __init__(self, name, count, parent):
        self.name = name
        self.count = count
        self.parent = parent
        self.children = {}
        self.node_link = None  # Link to the next node of the same name

    def increment(self, count):
        self.count += count


class FPTree:
    def __init__(self):
        self.root = FPTreeNode("null", 1, None)
        self.header_table = {}

    def add_transaction(self, transaction, count):
        current_node = self.root
        for item in transaction:
            if item in current_node.children:
                current_node.children[item].increment(count)
            else:
                new_node = FPTreeNode(item, count, current_node)
                current_node.children[item] = new_node
                # Update header table
                if item in self.header_table:
                    current = self.header_table[item]
                    while current.node_link:
                        current = current.node_link
                    current.node_link = new_node
                else:
                    self.header_table[item] = new_node
            current_node = current_node.children[item]

    def mine_patterns(self, min_support):
        patterns = []

        def mine_recursively(tree, suffix):
            # If the header table is empty, stop recursion
            if not tree.header_table:
                return

            # Iterate through each item in the header table
            for item in sorted(tree.header_table.keys()):
                # Add the current item to the suffix to form a new pattern
                pattern = suffix + [item]
                patterns.append(pattern)

                # Build the conditional pattern base
                conditional_base = []
                link_node = tree.header_table[item]
                while link_node:
                    path = []
                    parent = link_node.parent
                    while parent and parent.name != "null":
                        path.append(parent.name)
                        parent = parent.parent
                    path.reverse()
                    if path:
                        conditional_base.append((path, link_node.count))
                    link_node = link_node.node_link

                # Build the conditional FP-tree
                if conditional_base:
                    conditional_tree = FPTree()
                    item_counts = {}
                    for path, count in conditional_base:
                        for i in path:
                            item_counts[i] = item_counts.get(i, 0) + count
                    frequent_items = [
                        i for i, c in item_counts.items() if c >= min_support
                    ]
                    frequent_items.sort(key=lambda x: item_counts[x], reverse=True)
                    for path, count in conditional_base:
                        filtered_path = [i for i in path if i in frequent_items]
                        conditional_tree.add_transaction(filtered_path, count)

                    # Recurse on the conditional tree
                    mine_recursively(conditional_tree, pattern)

        mine_recursively(self, [])
        return patterns


def find_frequent_itemsets(dataset, min_support_count:int)->list [set[str]]:
    # Step 1: Calculate item frequencies
    item_counts = dataset.sum(axis=0).to_dict()

    # Filter items by minimum support
    frequent_items = {item: count for item, count in item_counts.items() if count >= min_support_count}

    # Sort items by descending support
    sorted_items = sorted(frequent_items.keys(), key=lambda x: frequent_items[x], reverse=True)

    # Step 2: Build FP-Tree
    fp_tree = FPTree()
    for _, row in dataset.iterrows():
        transaction = [item for item in sorted_items if row[item] == 1]
        fp_tree.add_transaction(transaction, 1)

    # Step 3: Mine the FP-Tree
    frequent_patterns = fp_tree.mine_patterns(min_support_count)
    return [set(pattern) for pattern in frequent_patterns]

def save_to_txt(filename, data):
    with open(filename, 'w') as file:
        for item in data:
            file.write(f"{item}\n")

def generate_rules(frequent_itemsets: list[set[str]], min_confidence, dataset) -> list[tuple[set[str], set[str]]]:
    rules = []

    def generate_subsets(itemset):
        subsets = []

        def helper(current, remaining):
            if not remaining:
                if current:
                    subsets.append(set(current))
                return
            helper(current + [remaining[0]], remaining[1:])
            helper(current, remaining[1:])

        helper([], list(itemset))
        return subsets

    def calculate_support(itemset, dataset):
        return dataset[list(itemset)].all(axis=1).sum()

    for itemset in frequent_itemsets:
        if len(itemset) < 2:
            continue
        subsets = generate_subsets(itemset)
        itemset_support = calculate_support(itemset, dataset)
        for antecedent in subsets:
            consequent = itemset - antecedent
            if antecedent and consequent:
                antecedent_support = calculate_support(antecedent, dataset)
                confidence = itemset_support / antecedent_support
                if confidence >= min_confidence:
                    rules.append((antecedent, consequent))
    return rules


def main():
    # Load the dataset
    dataset = pd.read_csv('../dist/adult_preprocessed.csv')  # Adjust the file path as necessary

    # Step 1: Find frequent itemsets
    min_support_count = 13000
    frequent_items = find_frequent_itemsets(dataset, min_support_count)
    print(f"Found {len(frequent_items)} frequent itemsets.")

    save_to_txt('../dist/freq_itemsets.txt', frequent_items)


    # Step 2: Generate association rules from frequent itemsets
    min_confidence = 0.95
    rules = generate_rules(frequent_items, min_confidence, dataset)
    print(f"Found {len(rules)} rules.")
    save_to_txt('../dist/rules.txt', rules)

if __name__ == "__main__":
    main()
