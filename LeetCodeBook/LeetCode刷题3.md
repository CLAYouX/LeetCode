[TOC]

## 对称二叉树

### 递归

如果同时满足下面的条件，两个树互为镜像：

- 它们的两个根结点具有相同的值
- 每棵树的右子树都与另一棵树的左子树镜像对称

<img src="https://assets.leetcode-cn.com/solution-static/101/101_fig2.PNG" width="600" height="330">

我们可以实现这样一个递归函数，通过**同步移动**两个指针的方法来遍历这棵树，$p$ 指针和 $q$ 指针一开始都指向这棵树的根，随后 $p$ 右移时，$q$ 左移，$p$ 左移时，$q$ 右移。每次检查当前 $p$ 和 $q$ 节点的值是否相等，如果相等再判断左右子树是否对称。

``` c++
class Solution {
public:
    bool check(TreeNode *p, TreeNode *q) {
        if (!p && !q) return true;
        if (!p || !q) return false;
        return p->val == q->val && check(p->left, q->right) && check(p->right, q->left);
    }

    bool isSymmetric(TreeNode* root) {
        return check(root, root);
    }
};
```

### 迭代

方法一中我们用递归的方法实现了对称性的判断，那么如何用迭代的方法实现呢？首先我们引入一个**队列**，这是把递归程序改写成迭代程序的常用方法。初始化时我们把根节点入队两次。每次提取两个结点并比较它们的值（队列中每两个连续的结点应该是相等的，而且它们的子树互为镜像），然后将两个结点的左右子结点按相反的顺序插入队列中。当队列为空时，或者我们检测到树不对称（即从队列中取出两个不相等的连续结点）时，该算法结束。

``` c++
class Solution {
public:
    bool check(TreeNode *u, TreeNode *v) {
        queue <TreeNode*> q;
        q.push(u); q.push(v);
        while (!q.empty()) {
            u = q.front(); q.pop();
            v = q.front(); q.pop();
            if (!u && !v) continue;
            if ((!u || !v) || (u->val != v->val)) return false;

            q.push(u->left); 
            q.push(v->right);

            q.push(u->right); 
            q.push(v->left);
        }
        return true;
    }

    bool isSymmetric(TreeNode* root) {
        return check(root, root);
    }
};
```

## 二叉树的层序遍历

### 广度优先搜索

我们可以想到最朴素的方法是用一个二元组 $(node, level)$ 来表示状态，它表示某个节点和它所在的层数，每个新进队列的节点的 `level` 值都是父亲节点的 `level` 值加一。最后根据每个点的 `level` 对点进行分类，分类的时候我们可以利用哈希表，维护一个以 `level` 为键，对应节点值组成的数组为值，广度优先搜索结束以后按键 `level` 从小到大取出所有值，组成答案返回即可。

我们可以用一种巧妙的方法修改广度优先搜索：

- 首先根元素入队
- 当队列不为空的时候
  - 求当前队列的长度 $s_i$
  - 依次从队列中取 $s_i$ 个元素进行拓展，然后进入下一次迭代

它和普通广度优先搜索的区别在于，普通广度优先搜索每次只取一个元素拓展，而这里每次取 $s_i$ 个元素。在上述过程中的第 $i$ 次迭代就得到了二叉树的第 $i$ 层的 $s_i$ 个元素。

``` C++
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector <vector <int>> ret;
        if (!root) {
            return ret;
        }

        queue <TreeNode*> q;
        q.push(root);
        while (!q.empty()) {
            int currentLevelSize = q.size();
            ret.push_back(vector <int> ());
            for (int i = 1; i <= currentLevelSize; ++i) {
                auto node = q.front(); q.pop();
                ret.back().push_back(node->val);
                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
            }
        }
        
        return ret;
    }
};
```

## 二叉树的锯齿形层序遍历

为了满足题目要求的返回值为「先从左往右，再从右往左」交替输出的锯齿形，我们可以利用**双端队列**的数据结构来维护当前层节点值输出的顺序。

双端队列是一个可以在队列任意一端插入元素的队列。在广度优先搜索遍历当前层节点拓展下一层节点的时候我们仍然从左往右按顺序拓展，但是对当前层节点的存储我们维护一个变量 $\textit{isOrderLeft}$ 记录是从左至右还是从右至左的：

- 如果从左至右，我们每次将被遍历到的元素插入至双端队列的末尾。
- 如果从右至左，我们每次将被遍历到的元素插入至双端队列的头部。

``` c++
class Solution {
public:
    vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
        vector<vector<int>> ans;
        if (!root) {
            return ans;
        }

        queue<TreeNode*> nodeQueue;
        nodeQueue.push(root);
        bool isOrderLeft = true;

        while (!nodeQueue.empty()) {
            deque<int> levelList;
            int size = nodeQueue.size();
            for (int i = 0; i < size; ++i) {
                auto node = nodeQueue.front();
                nodeQueue.pop();
                if (isOrderLeft) {
                    levelList.push_back(node->val);
                } else {
                    levelList.push_front(node->val);
                }
                if (node->left) {
                    nodeQueue.push(node->left);
                }
                if (node->right) {
                    nodeQueue.push(node->right);
                }
            }
            ans.emplace_back(vector<int>{levelList.begin(), levelList.end()});
            isOrderLeft = !isOrderLeft;
        }

        return ans;
    }
};
```

## 二叉树最大深度

### 深度优先搜索

如果我们知道了左子树和右子树的最大深度 $l$ 和 $r$，那么该二叉树的最大深度即为：
$$
max(l,r)+1
$$
而左子树和右子树的最大深度又可以以同样的方式进行计算。因此我们可以用「深度优先搜索」的方法来计算二叉树的最大深度。具体而言，在计算当前二叉树的最大深度时，可以先递归计算出其左子树和右子树的最大深度，然后在 $O(1)$ 时间内计算出当前二叉树的最大深度。递归在访问到空节点时退出。

``` c++
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if (root == nullptr) return 0;
        return max(maxDepth(root->left), maxDepth(root->right)) + 1;
    }
};
```

### 广度优先搜索

每次拓展下一层的时候，我们需要将队列里的所有节点都拿出来进行拓展，这样能保证每次拓展完的时候队列里存放的是当前层的所有节点，即我们是一层一层地进行拓展，最后我们用一个变量 $\textit{ans}$ 来维护拓展的次数，该二叉树的最大深度即为 $\textit{ans}$。

```c++
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if (root == nullptr) return 0;
        queue<TreeNode*> Q;
        Q.push(root);
        int ans = 0;
        while (!Q.empty()) {
            int sz = Q.size();
            while (sz > 0) {
                TreeNode* node = Q.front();Q.pop();
                if (node->left) Q.push(node->left);
                if (node->right) Q.push(node->right);
                sz -= 1;
            }
            ans += 1;
        } 
        return ans;
    }
};

## 从前序与中序遍历序列构造二叉树

### 递归

对于任意一颗树而言，前序遍历的形式总是：

`[ 根节点, [左子树的前序遍历结果], [右子树的前序遍历结果] ]`

即根节点总是前序遍历中的第一个节点。而中序遍历的形式总是：

`[ [左子树的中序遍历结果], 根节点, [右子树的中序遍历结果] ]`

只要我们在中序遍历中定位到根节点，那么我们就可以分别知道左子树和右子树中的节点数目。由于同一颗子树的前序遍历和中序遍历的长度显然是相同的，因此我们就可以对应到前序遍历的结果中，对上述形式中的所有**左右括号**进行定位。

**细节：**在中序遍历中对根节点进行定位时，一种简单的方法是直接扫描整个中序遍历的结果并找出根节点，但这样做的时间复杂度较高。我们可以考虑使用哈希表来帮助我们快速地定位根节点。对于哈希映射中的每个键值对，键表示一个元素（节点的值），值表示其在中序遍历中的出现位置。在构造二叉树的过程之前，我们可以对中序遍历的列表进行一遍扫描，就可以构造出这个哈希映射。在此后构造二叉树的过程中，我们就只需要 $O(1)$ 的时间对根节点进行定位了。

``` c++
class Solution {
private:
    unordered_map<int, int> index;

public:
    TreeNode* myBuildTree(const vector<int>& preorder, const vector<int>& inorder, 
                          int preorder_left, int preorder_right, int inorder_left, int inorder_right) {
        
        if (preorder_left > preorder_right) {
            return nullptr;
        }
        
        // 前序遍历中的第一个节点就是根节点
        int preorder_root = preorder_left;
        // 在中序遍历中定位根节点
        int inorder_root = index[preorder[preorder_root]];
        
        // 先把根节点建立出来
        TreeNode* root = new TreeNode(preorder[preorder_root]);
        // 得到左子树中的节点数目
        int size_left_subtree = inorder_root - inorder_left;
        // 递归地构造左子树，并连接到根节点
        // 先序遍历中「从 左边界+1 开始的 size_left_subtree」个元素就对应了中序遍历中「从 左边界 开始到 根节点定位-1」的元素
        root->left = myBuildTree(preorder, inorder, preorder_left + 1, preorder_left + size_left_subtree, 
                                 inorder_left, inorder_root - 1);
        // 递归地构造右子树，并连接到根节点
        // 先序遍历中「从 左边界+1+左子树节点数目 开始到 右边界」的元素就对应了中序遍历中「从 根节点定位+1 到 右边界」的元素
        root->right = myBuildTree(preorder, inorder, preorder_left + size_left_subtree + 1, preorder_right, 
                                  inorder_root + 1, inorder_right);
        return root;
    }

    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        int n = preorder.size();
        // 构造哈希映射，帮助我们快速定位根节点
        for (int i = 0; i < n; ++i) {
            index[inorder[i]] = i;
        }
        return myBuildTree(preorder, inorder, 0, n - 1, 0, n - 1);
    }
};
```

## 迭代

迭代法是一种非常巧妙的实现方法。

对于前序遍历中的任意两个连续节点 $u$ 和 $v$，根据前序遍历的流程，我们可以知道 $u$ 和 $v$ 只有两种可能的关系：

- $v$ 是 $u$ 的左儿子。这是因为在遍历到 $u$ 之后，下一个遍历的节点就是 $u$ 的左儿子，即 $v$；
- $u$ 没有左儿子，并且 $v$ 是 $u$ 的某个祖先节点（或者 $u$ 本身）的右儿子。如果 $u$ 没有左儿子，那么下一个遍历的节点就是 $u$ 的右儿子。如果 $u$ 没有右儿子，我们就会向上回溯，直到遇到第一个有右儿子（且 $u$ 不在它的右儿子的子树中）的节点 $u_a$，那么 $v$ 就是 $u_a$ 的右儿子。

以树

```
        3
       / \
      9  20
     /  /  \
    8  15   7
   / \
  5  10
 /
4
```

为例。它的前序遍历和中序遍历分别为：

```
preorder = [3, 9, 8, 5, 4, 10, 20, 15, 7]
inorder = [4, 5, 8, 10, 9, 3, 15, 20, 7]
```

我们用一个栈 `stack` 来维护**当前节点的所有还没有考虑过右儿子的祖先节点**，栈顶就是当前节点。也就是说，只有在栈中的节点才可能连接一个新的右儿子。同时，我们用一个指针 `index` 指向中序遍历的某个位置，初始值为 $0$。`index` 对应的节点是**当前节点不断往左走达到的最终节点**，这也是符合中序遍历的，它的作用在下面的过程中会有所体现。

首先我们将根节点 $3$ 入栈，再初始化 `index` 所指向的节点为 $4$，随后对于前序遍历中的每个节点，我们依次判断它是栈顶节点的左儿子，还是栈中某个节点的右儿子。

- 我们遍历 $9$。$9$ 一定是栈顶节点 $3$ 的左儿子。我们使用反证法，假设 $9$ 是 $3$ 的右儿子，那么 $3$ 没有左儿子，`index` 应该恰好指向 $3$，但实际上为 $4$，因此产生了矛盾。所以我们将 $9$ 作为 $3$ 的左儿子，并将 $9$ 入栈。

  - `stack = [3, 9]`
  - `index -> inorder[0] = 4`

- 我们遍历 $8$，$5$ 和 $4$。同理可得它们都是上一个节点（栈顶节点）的左儿子，所以它们会依次入栈。
  - `stack = [3, 9, 8, 5, 4]`
  - `index -> inorder[0] = 4`

- 我们遍历 $10$，这时情况就不一样了。我们发现 `index` 恰好指向当前的栈顶节点 $4$，也就是说 $4$ 没有左儿子，那么 $10$ 必须为栈中某个节点的右儿子。那么如何找到这个节点呢？栈中的节点的顺序和它们在前序遍历中出现的顺序是一致的，而且每一个节点的右儿子都还没有被遍历过，那么**这些节点的顺序和它们在中序遍历中出现的顺序一定是相反的**。

  > 这是因为栈中的任意两个相邻的节点，前者都是后者的某个祖先。并且我们知道，栈中的任意一个节点的右儿子还没有被遍历过，说明后者一定是前者左儿子的子树中的节点，那么后者就先于前者出现在中序遍历中。

  因此我们可以把 `index` 不断向右移动，并与栈顶节点进行比较。如果 `index` 对应的元素恰好等于栈顶节点，那么说明我们在中序遍历中找到了栈顶节点，所以将 `index` 增加 $1$ 并弹出栈顶节点，直到 `index` 对应的元素不等于栈顶节点。按照这样的过程，我们弹出的最后一个节点 $x$ 就是 $10$ 的双亲节点，**这是因为 $10$ 出现在了 $x$ 与 $x$ 在栈中的下一个节点的中序遍历之间**，因此 $10$ 就是 $x$ 的右儿子。

  回到我们的例子，我们会依次从栈顶弹出 $4$，$5$ 和 $8$，并且将 `index` 向右移动了三次。我们将 $10$ 作为最后弹出的节点 $8$ 的右儿子，并将 $10$ 入栈。

  - `stack=[3, 8, 10]`
  - `index -> inorder[3] = 10`

- 我们遍历 $20$。同理，`index` 恰好指向当前栈顶节点 $10$，那么我们会依次从栈顶弹出 $10$，$9$ 和 $3$，并且将 `index` 向右移动了三次。我们将 $20$ 作为最后弹出的节点 $3$ 的右儿子，并将 $20$ 入栈。

  - `stack=[20]`
  - `index -> inorder[6] = 15`

- 我们遍历 $15$，将 $15$ 作为栈顶节点 $20$ 的左儿子，并将 $15$ 入栈。
  - `stack=[15]`
  - `index -> inorder[6] = 15`

- 我们遍历 $7$。`index` 恰好指向当前栈顶节点 $15$，那么我们会依次从栈顶弹出 $15$ 和 $20$，并且将 `index` 向右移动了两次。我们将 $7$ 作为最后弹出的节点 $20$ 的右儿子，并将 $7$ 入栈。

  - `stack=[7]`
  - `index -> inorder[8] = 7`

**算法步骤：**

1. 用一个栈和一个指针辅助进行二叉树的构造。初始时栈中存放了根节点（前序遍历的第一个节点），指针指向中序遍历的第一个节点；
2. 依次枚举前序遍历中除了第一个节点以外的每个节点。如果 `index` 恰好指向栈顶节点，那么我们不断地弹出栈顶节点并向右移动 `index`，并将当前节点作为最后一个弹出的节点的右儿子；如果 `index` 和栈顶节点不同，我们将当前节点作为栈顶节点的左儿子；
3. 无论是哪一种情况，我们最后都将当前的节点入栈。

``` c++
class Solution {
public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        if (!preorder.size()) {
            return nullptr;
        }
        TreeNode* root = new TreeNode(preorder[0]);
        stack<TreeNode*> stk;
        stk.push(root);
        int inorderIndex = 0;
        for (int i = 1; i < preorder.size(); ++i) {
            int preorderVal = preorder[i];
            TreeNode* node = stk.top();
            if (node->val != inorder[inorderIndex]) {
                node->left = new TreeNode(preorderVal);
                stk.push(node->left);
            }
            else {
                while (!stk.empty() && stk.top()->val == inorder[inorderIndex]) {
                    node = stk.top();
                    stk.pop();
                    ++inorderIndex;
                }
                node->right = new TreeNode(preorderVal);
                stk.push(node->right);
            }
        }
        return root;
    }
};
```

