[TOC]
## Pow(x,n)
### 快速幂-递归
快速幂本质是分治算法。举个例子，如果我们要计算$x^{64}$，我们可以按照：
$$
x \rightarrow x^2 \rightarrow x^4 \rightarrow x^8 \rightarrow x^{16} \rightarrow x^{32} \rightarrow x^{64}
$$
的顺序，从 $x$ 开始，每次直接把上一次的结果进行平方，计算 66 次就可以得到 $x^64$ 的值。
再举一个例子，如果我们要计算 $x^{77}$ 我们可以按照：
$$
x \rightarrow x^2 \rightarrow x^4 \rightarrow x^9 \rightarrow x^{19} \rightarrow x^{38} \rightarrow x^{77}
$$
的顺序。
直接从左到右进行推导看上去很困难，因为在每一步中，我们不知道在将上一次的结果平方之后，还需不需要额外乘 $x$。但如果我们从右往左看，分治的思想就十分明显了：

- 当我们要计算 $x^n$ 时，我们可以先递归地计算出 $y = x^{\lfloor n/2 \rfloor}$;
- 根据递归计算的结果，如果 $n$ 为偶数，那么 $x^n = y^2$；如果 $n$ 为奇数，那么 $x^n = y^2 \times x$;
- 递归的边界为 $n = 0$，任意数的 $0$ 次方均为 $1$。
``` c++
class Solution {
public:
    double quickMul(double x, long long N) {
        if (N == 0) {
            return 1.0;
        }
        double y = quickMul(x, N / 2);
        return N % 2 == 0 ? y * y : y * y * x;
    }

    double myPow(double x, int n) {
        long long N = n;
        return N >= 0 ? quickMul(x, N) : 1.0 / quickMul(x, -N);
    }
};
```
### 快速幂-迭代
还是以 $ x^{77} $ 作为例子：
$$
x \rightarrow x^2 \rightarrow x^4 \rightarrow +x^9 \rightarrow +x^{19} \rightarrow x^{38} \rightarrow +x^{77}
$$
并且把需要额外乘 $x$ 的步骤打上了 $+$ 标记。可以发现：
- $x^{38} \rightarrow +x^{77}$中额外乘的 $x$ 在 $x^{77}$ 贡献了 $x$
- $+x^9 \rightarrow +x^{19}$中额外乘的 $x$ 在 $x^{77}$ 贡献了 $x^{2^2} = x^4$
- $x^4 \rightarrow +x^9$中额外乘的 $x$ 在 $x^{77}$ 贡献了 $x^{2^3} = x^8$
- 最初的 $x$ 在之后被平方了 $6$ 次，因此在$x^{77}$ 贡献了 $x^{2^6} = x^{64}$
把这些贡献相乘， $x^2 \times x^4 \times x^8 \times x^{64}$ 恰好等于 $x^{77}$。而这些贡献的指数部分又是什么呢？它们都是 $2$ 的幂次，这是因为每个额外乘的 $x$ 在之后都会被平方若干次。而这些指数 $1，4，8 和 64$，恰好就对应了 $77$ 的二进制表示 $(1001101)_2$(1001101) 中的每个 $1$。
我们从 $x$ 开始不断地进行平方，得到 $x^2, x^4, x^8, x^{16}, \cdots$，如果 $n$ 的第 $k$ 个（从右往左，从 $0$ 开始计数）二进制位为 $1$，那么我们就将对应的贡献 $x^{2^k}$x 计入答案。
``` c++
class Solution {
public:
    double quickMul(double x, long long N) {
        double ans = 1.0;
        // 贡献的初始值为 x
        double x_contribute = x;
        // 在对 N 进行二进制拆分的同时计算答案
        while (N > 0) {
            if (N % 2 == 1) {
                // 如果 N 二进制表示的最低位为 1，那么需要计入贡献
                ans *= x_contribute;
            }
            // 将贡献不断地平方
            x_contribute *= x_contribute;
            // 舍弃 N 二进制表示的最低位，这样我们每次只要判断最低位即可
            N /= 2;
        }
        return ans;
    }

    double myPow(double x, int n) {
        long long N = n;
        return N >= 0 ? quickMul(x, N) : 1.0 / quickMul(x, -N);
    }
};
```
## N皇后
### 基于集合的回溯
为了判断一个位置所在的列和两条斜线上是否已经有皇后，使用三个集合 $columns、diagonals_1$  和 $diagonals_2$ 分别记录每一列以及两个方向的每条斜线上是否有皇后。
方向一的斜线为从左上到右下方向，同一条斜线上的每个位置满足行下标与列下标之差相等。
<img src="https://assets.leetcode-cn.com/solution-static/51/1.png" width="700" height="400">
方向二的斜线为从右上到左下方向，同一条斜线上的每个位置满足行下标与列下标之和相等。
<img src="https://assets.leetcode-cn.com/solution-static/51/2.png" width="700" height="400">
每次放置皇后时，对于每个位置判断其是否在三个集合中，如果三个集合都不包含当前位置，则当前位置是可以放置皇后的位置。

``` c++
class Solution {
public:
    vector<vector<string>> solveNQueens(int n) {
        auto solutions = vector<vector<string>>();
        auto queens = vector<int>(n, -1);
        auto columns = unordered_set<int>();
        auto diagonals1 = unordered_set<int>();
        auto diagonals2 = unordered_set<int>();
        backtrack(solutions, queens, n, 0, columns, diagonals1, diagonals2);
        return solutions;
    }

    void backtrack(vector<vector<string>> &solutions, vector<int> &queens, int n, int row, unordered_set<int> &columns, unordered_set<int> &diagonals1, unordered_set<int> &diagonals2) {
        if (row == n) {
            vector<string> board = generateBoard(queens, n);
            solutions.push_back(board);
        } else {
            for (int i = 0; i < n; i++) {
                if (columns.find(i) != columns.end()) {
                    continue;
                }
                int diagonal1 = row - i;
                if (diagonals1.find(diagonal1) != diagonals1.end()) {
                    continue;
                }
                int diagonal2 = row + i;
                if (diagonals2.find(diagonal2) != diagonals2.end()) {
                    continue;
                }
                queens[row] = i;
                columns.insert(i);
                diagonals1.insert(diagonal1);
                diagonals2.insert(diagonal2);
                backtrack(solutions, queens, n, row + 1, columns, diagonals1, diagonals2);
                queens[row] = -1;
                columns.erase(i);
                diagonals1.erase(diagonal1);
                diagonals2.erase(diagonal2);
            }
        }
    }

    vector<string> generateBoard(vector<int> &queens, int n) {
        auto board = vector<string>();
        for (int i = 0; i < n; i++) {
            string row = string(n, '.');
            row[queens[i]] = 'Q';
            board.push_back(row);
        }
        return board;
    }
};
```
### 基于位运算的回溯
如果利用位运算记录皇后的信息，就可以将记录皇后信息的空间复杂度从 $O(N)$ 降到 $O(1)$。
具体做法是，使用三个整数 $columns、diagonals_1$  和 $diagonals_2$ 分别记录每一列以及两个方向的每条斜线上是否有皇后，每个整数有 $N$ 个二进制位。棋盘的每一列对应每个整数的二进制表示中的一个数位，其中棋盘的最左列对应每个整数的最低二进制位，最右列对应每个整数的最高二进制位。我们用 `0 代表可以放置皇后的位置`，`1 代表不能放置皇后的位置`。三个整数的计算方法为：
- 初始时，三个整数的值都等于 $0$，表示没有放置任何皇后；
- 在当前行放置皇后，如果皇后放置在第 $i$ 列，则将三个整数的第 $i$ 个二进制位（指从低到高的第 $i$ 个二进制位）的值设为 $1$；
- 进入下一行时，$columns$ 的值保持不变，$diagonals_1$左移一位，$diagonals_2$右移一位，由于棋盘的最左列对应每个整数的最低二进制位，即每个整数的最右二进制位，因此对整数的移位操作方向和对棋盘的移位操作方向相反（对棋盘的移位操作方向是 $diagonals_1$右移一位，$diagonals_2$左移一位）

每次放置皇后时，三个整数的按位或运算的结果即为不能放置皇后的位置，其余位置即为可以放置皇后的位置。可以通过 $(2^n-1)~\&~(\sim(columns | diagonals_1 | diagonals_2))$ 得到可以放置皇后的位置（该结果的值为 `1 的位置表示可以放置皇后的位置`），然后遍历这些位置，尝试放置皇后并得到可能的解。遍历可以放置皇后的位置时，可以利用以下两个按位与运算的性质：
- $x \& (−x)$ 可以获得 $x$ 的二进制表示中的最低位的 $1$ 的位置；
- $x \& (x−1)$ 可以将 $x$ 的二进制表示中的最低位的 $1$ 置成 $0$。

具体做法是，每次获得可以放置皇后的位置中的最低位，并将该位的值置成 $0$，尝试在该位置放置皇后。这样即可遍历每个可以放置皇后的位置。
``` c++
class Solution {
public:
    vector<vector<string>> solveNQueens(int n) {
        auto solutions = vector<vector<string>>();
        auto queens = vector<int>(n, -1);
        solve(solutions, queens, n, 0, 0, 0, 0);
        return solutions;
    }

    void solve(vector<vector<string>> &solutions, vector<int> &queens, int n, int row, int columns, int diagonals1, int diagonals2) {
        if (row == n) {
            auto board = generateBoard(queens, n);
            solutions.push_back(board);
        } else {
            int availablePositions = ((1 << n) - 1) & (~(columns | diagonals1 | diagonals2));
            while (availablePositions != 0) {
                int position = availablePositions & (-availablePositions);
                availablePositions = availablePositions & (availablePositions - 1);
                int column = __builtin_ctz(position); // 末尾0的个数
                queens[row] = column;
                solve(solutions, queens, n, row + 1, columns | position, (diagonals1 | position) >> 1, (diagonals2 | position) << 1);
                queens[row] = -1;
            }
        }
    }

    vector<string> generateBoard(vector<int> &queens, int n) {
        auto board = vector<string>();
        for (int i = 0; i < n; i++) {
            string row = string(n, '.');
            row[queens[i]] = 'Q';
            board.push_back(row);
        }
        return board;
    }
};
```
## 最大子序和
### 动态规划
我们用 $f(i)$ 代表以第 $i$ 个数结尾的 **连续子数组的最大和**，可以写出这样的动态规划转移方程：
$$
f[i] = max(f[i-1]+nums[i], nums[i])
$$
考虑到 $f(i)$ 只和 $f(i−1)$ 相关，于是我们可以只用一个变量 $pre$ 来维护对于当前 $f(i)$ 的 $f(i−1)$ 的值是多少，从而让空间复杂度降低到 $O(1)$，这有点类似**滚动数组**的思想。
``` c++
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int pre = 0, maxAns = nums[0];
        for (const auto &x: nums) {
            pre = max(pre + x, x);
            maxAns = max(maxAns, pre);
        }
        return maxAns;
    }
};
```
### 分治
#### 思路
这个分治方法类似于**线段树求解最长公共上升子序列问题**的 $pushUp$ 操作。
我们定义一个操作 $get(a, l, r)$ 表示查询 $a$ 序列 $[l,r]$ 区间内的最大子段和，那么最终我们要求的答案就是 $get(nums, 0, nums.size() - 1)$。对于一个区间 $[l,r]$，我们取 $m = \lfloor \frac{l + r}{2} \rfloor$，对区间 $[l,m]$ 和 $[m+1,r]$ 分治求解。当递归逐层深入直到区间长度缩小为 $1$ 的时候，递归**开始回升**。这个时候我们考虑如何通过 $[l,m]$ 区间的信息和 $[m+1,r]$ 区间的信息合并成区间 $[l,r]$ 的信息。
对于一个区间 $[l,r]$，我们可以维护四个量：
- $lSum$ 表示 $[l,r]$ 内以 $l$ 为左端点的最大子段和
- $rSum$ 表示 $[l,r]$ 内以 $r$ 为右端点的最大子段和
- $mSum$ 表示 $[l,r]$ 内的最大子段和
- $iSum$ 表示 $[l,r]$ 的区间和
对于长度为 $1$ 的区间 $[i, i]$，四个量的值都和 $nums[i]$ 相等。对于长度大于 $1$ 的区间：
- 首先最好维护的是 $iSum$，区间 $[l,r]$ 的 $iSum$ 就等于**左子区间**的 $iSum$ 加上**右子区间**的 $iSum$。
- 对于 $[l,r]$ 的 $lSum$，存在两种可能，它要么等于**左子区间**的 $lSum$，要么等于**左子区间**的 $iSum$ 加上**右子区间**的 $lSum$，二者取大。
- 对于 $[l,r]$ 的 $rSum$，同理，它要么等于**右子区间**的 $rSum$，要么等于**右子区间**的 $iSum$ 加上**左子区间**的 $rSum$，二者取大。
- 当计算好上面的三个量之后，就很好计算 $[l,r]$ 的 $mSum$ 了。我们可以考虑 $[l,r]$ 的 $mSum$ 对应的区间是否跨越 $m$——它可能不跨越 $m$，也就是说 $[l,r]$ 的 $mSum$ 可能是**左子区间**的 $mSum$ 和 **右子区间**的 $mSum$ 中的一个；它也可能跨越 $m$，可能是**左子区间**的 $rSum$ 和**右子区间**的 $lSum$ 求和。三者取大。
``` c++
class Solution {
public:
    struct Status {
        int lSum, rSum, mSum, iSum;
    };

    Status pushUp(Status l, Status r) {
        int iSum = l.iSum + r.iSum;
        int lSum = max(l.lSum, l.iSum + r.lSum);
        int rSum = max(r.rSum, r.iSum + l.rSum);
        int mSum = max(max(l.mSum, r.mSum), l.rSum + r.lSum);
        return (Status) {lSum, rSum, mSum, iSum};
    };

    Status get(vector<int> &a, int l, int r) {
        if (l == r) {
            return (Status) {a[l], a[l], a[l], a[l]};
        }
        int m = (l + r) >> 1;
        Status lSub = get(a, l, m);
        Status rSub = get(a, m + 1, r);
        return pushUp(lSub, rSub);
    }

    int maxSubArray(vector<int>& nums) {
        return get(nums, 0, nums.size() - 1).mSum;
    }
};
```
#### 复杂度分析
- 时间复杂度：假设我们把递归的过程看作是一颗二叉树的先序遍历，那么这颗二叉树的深度的渐进上界为 $O(logn)$，这里的总时间相当于遍历这颗二叉树的所有节点，故总时间的渐进上界是 $O(\sum_{i=1}^{\log n} 2^{i-1})=O(n)$
- 空间复杂度：递归会使用 $O(logn)$ 的栈空间
## 螺旋矩阵
### 模拟
可以模拟螺旋矩阵的路径。初始位置是矩阵的左上角，初始方向是向右，当路径超出界限或者进入之前访问过的位置时，顺时针旋转，进入下一个方向。
由于矩阵中的每个元素都被访问一次，因此路径的长度即为矩阵中的元素数量，当路径的长度达到矩阵中的元素数量时即为完整路径，将该路径返回。
``` c++
class Solution {
private:
    static constexpr int directions[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        if (matrix.size() == 0 || matrix[0].size() == 0) {
            return {};
        }
        
        int rows = matrix.size(), columns = matrix[0].size();
        vector<vector<bool>> visited(rows, vector<bool>(columns));
        int total = rows * columns;
        vector<int> order(total);

        int row = 0, column = 0;
        int directionIndex = 0;
        for (int i = 0; i < total; i++) {
            order[i] = matrix[row][column];
            visited[row][column] = true;
            int nextRow = row + directions[directionIndex][0], nextColumn = column + directions[directionIndex][1];
            if (nextRow < 0 || nextRow >= rows || nextColumn < 0 || nextColumn >= columns || visited[nextRow][nextColumn]) {
                directionIndex = (directionIndex + 1) % 4;
            }
            row += directions[directionIndex][0];
            column += directions[directionIndex][1];
        }
        return order;
    }
};
```
### 按层模拟
可以将矩阵看成若干层，首先输出最外层的元素，其次输出次外层的元素，直到输出最内层的元素。对于每层，从左上方开始以顺时针的顺序遍历所有元素。
<img src="https://assets.leetcode-cn.com/solution-static/54/54_fig1.png" width="800" height="400">

``` c++
class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        if (matrix.size() == 0 || matrix[0].size() == 0) {
            return {};
        }

        int rows = matrix.size(), columns = matrix[0].size();
        vector<int> order;
        int left = 0, right = columns - 1, top = 0, bottom = rows - 1;
        while (left <= right && top <= bottom) {
            for (int column = left; column <= right; column++) {
                order.push_back(matrix[top][column]);
            }
            for (int row = top + 1; row <= bottom; row++) {
                order.push_back(matrix[row][right]);
            }
            if (left < right && top < bottom) {
                for (int column = right - 1; column > left; column--) {
                    order.push_back(matrix[bottom][column]);
                }
                for (int row = bottom; row > top; row--) {
                    order.push_back(matrix[row][left]);
                }
            }
            left++;
            right--;
            top++;
            bottom--;
        }
        return order;
    }
};
```
## 合并区间
首先，我们将列表中的区间按照左端点升序排序。然后我们将第一个区间加入 $merged$ 数组中，并按顺序依次考虑之后的每个区间：
- 如果当前区间的左端点在数组 $merged$ 中最后一个区间的右端点之后，那么它们不会重合，我们可以直接将这个区间加入数组 $merged$ 的末尾；
- 否则，它们重合，我们需要用当前区间的右端点更新数组 $merged$ 中最后一个区间的右端点，将其置为二者的较大值。
``` c++
class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        if (intervals.size() == 0) {
            return {};
        }
        sort(intervals.begin(), intervals.end());
        vector<vector<int>> merged;
        for (int i = 0; i < intervals.size(); ++i) {
            int L = intervals[i][0], R = intervals[i][1];
            if (!merged.size() || merged.back()[1] < L) {
                merged.push_back({L, R});
            }
            else {
                merged.back()[1] = max(merged.back()[1], R);
            }
        }
        return merged;
    }
};
```

## 插入区间

### 模拟

在给定区间集合**互不重叠**的前提下，当我们需要插入一个新的区间$S=[left,right]$时，我们只需要：

- 找出所有区间$S$重叠的区间集合$X'$;
- 将$X'$中的所有区间连带上区间$S$合并为一个大区间
- 最终的答案即为不与$X'$重叠的区间以及合并后的大区间

<img src="https://assets.leetcode-cn.com/solution-static/57/1.png" width="800" height="400">

并且，在给定的区间集合已经按照左端点排序的前提下，所有与区间 $S$ 重叠的区间在数组 $\textit{intervals}$ 中下标范围是连续的，因此我们可以对所有的区间进行一次遍历，就可以找到这个连续的下标范围。

当我们遍历区间 $[l_i, r_i]$时：

- 如果 $r_i < \textit{left}$，说明 $[l_i, r_i]$ 与 $S$ 不重叠并且在其左侧，我们可以直接将 $[l_i, r_i]$加入答案；
- 如果 $l_i > \textit{right}$，说明 $[l_i, r_i]$ 与 $S$ 不重叠并且在其右侧，我们可以直接将 $[l_i, r_i]$ 加入答案；
- 如果上面两种情况均不满足，说明$[l_i, r_i]$ 与 $S$ 重叠。此时我们需要将 $S$ 与 $[l_i,r_i]$ 合并，即将 $S$ 更新为其与 $[l_i,r_i]$ 的并集：

$$
[min(l_1,l_2), max(r_1,r_2)]
$$

那么我们应当在什么时候将区间 $S$ 加入答案呢？由于我们需要保证答案也是按照左端点排序的，因此当我们遇到第一个 满足 $l_i > \textit{right}$ 的区间时，说明以后遍历到的区间不会与 $S$ 重叠，并且它们左端点一定会大于 $S$ 的左端点。此时我们就可以将 $S$ 加入答案。特别地，如果不存在这样的区间，我们需要在遍历结束后，将 $S$ 加入答案。
``` c++
class Solution {
public:
    vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval) {
        int left = newInterval[0];
        int right = newInterval[1];
        bool placed = false;
        vector<vector<int>> ans;
        for (const auto& interval: intervals) {
            if (interval[0] > right) {
                // 在插入区间的右侧且无交集
                if (!placed) {
                    ans.push_back({left, right});
                    placed = true;                    
                }
                ans.push_back(interval);
            }
            else if (interval[1] < left) {
                // 在插入区间的左侧且无交集
                ans.push_back(interval);
            }
            else {
                // 与插入区间有交集，计算它们的并集
                left = min(left, interval[0]);
                right = max(right, interval[1]);
            }
        }
        if (!placed) {
            ans.push_back({left, right});
        }
        return ans;
    }
};
```

## 排列序列(第K个排列)

### 数学+缩小问题规模

#### 思路

对于给定的 $n$ 和 $k$，我们不妨从左往右确定第 $k$ 个排列中的每一个位置上的元素到底是什么。

我们首先确定排列中的首个元素 $a_1$ 根据上述的结论，我们可以知道：

- 以 $1$ 为 $a_1$ 的排列一共有 $(n-1)!$ 个；
- 以 $2$ 为 $a_1$ 的排列一共有 $(n-1)!$ 个；
- $\cdots$
- 以 $n$ 为 $a_1$ 的排列一共有 $(n-1)!$ 个；

由于我们需要求出从小到大的第 $k$ 个排列，因此：

- 如果 $k \leq (n-1)!$，我们就可以确定排列的首个元素为 $1$；
- 如果 $(n-1)! < k \leq 2 \cdot (n-1)!$，我们就可以确定排列的首个元素为 $2$；
- $\cdots$
- 如果 $(n-1) \cdot (n-1)! < k \leq n \cdot (n-1)!$，我们就可以确定排列的首个元素为 $n$。

因此，第 $k$ 个排列的首个元素就是：
$$
a_1 = ⌊\frac{k-1}{(n-1)!}⌋+1
$$
当我们确定了 $a_1$ 后，考虑以 $a_1$ 为首个元素的所有排列：

- 以 $[1,n] \backslash a_1$ 最小的元素为 $a_2$ 的排列一共有 $(n-2)!$ 个；
- 以 $[1,n] \backslash a_1$ 次小的元素为 $a_2$ 的排列一共有 $(n-2)!$ 个；
- $\cdots$
- 以 $[1,n] \backslash a_1$ 最大的元素为 $a_2$ 的排列一共有 $(n-2)!$ 个；

其中，$[1,n] \backslash a_1$ 表示包含 $1,2,\cdots,n$ 中出去 $a_1$ 以外元素的集合。这些排列从编号 $(a_1-1) \cdot (n-1)!$开始，到 $a_1 \cdot (n-1)!$ 结束，因此第 $k$ 个排列实际上就对应着这其中的第
$$
k'=(k-1)mod(n-1)!+1
$$
个排列。这样一来，我们就把原问题转化成了一个完全相同但规模减少 $1$ 的子问题：

> 求$[1,n] \backslash a_1$ 这 $n-1$ 个元素组成的排列中，第 $k'$ 小的排列

#### 算法

设第 $k$ 个排列为 $a_1, a_2, \cdots, a_n$， 我们从左往右地确定每一个元素 $a_i$。用数组 $valid$ 记录每一个元素是否被使用过。

从小到大枚举 $i$：

- 已经使用过了 $i-1$ 个元素，剩余 $n-i+1$ 个元素未使用过，每一个元素作为 $a_i$ 都对应着 $(n-i)!$ 个排列，总计 $(n-i+1)!$ 个排列；
- 因此在第 $k$ 个排列中，$a_i$ 即为剩余未使用过的元素中第 $⌊\frac{k-1}{(n-1)!}⌋+1$ 小的元素
- 在确定了 $a_i$ 后，这 $n-i+1$ 个元素的第 $k$ 个排列， 就等于 $a_i$ 之后跟着剩余 $n-i$ 个元素的第 $(k-1)mod(n-1)!+1$ 个排列

在实际的代码中，我们可以首先将 $k$ 的值减少 $1$，这样可以减少运算，降低代码出错的概率。实际上，这相当于我们将所有的排列从 $0$ 开始进行编号。

``` c++
class Solution {
public:
    string getPermutation(int n, int k) {
        vector<int> factorial(n);
        factorial[0] = 1;
        for (int i = 1; i < n; ++i) {
            factorial[i] = factorial[i - 1] * i;
        }

        --k;
        string ans;
        vector<int> valid(n + 1, 1);
        for (int i = 1; i <= n; ++i) {
            int order = k / factorial[n - i] + 1;
            for (int j = 1; j <= n; ++j) {
                order -= valid[j];
                if (!order) {
                    ans += (j + '0');
                    valid[j] = 0;
                    break;
                }
            }
            k %= factorial[n - i];
        }   
        return ans;     
    }
};
```

## 旋转链表

记给定链表的长度为 $n$，注意到当向右移动的次数 $k \geq n$ 时，我们仅需要向右移动 $k \bmod n$ 次即可。因为每 $n$ 次移动都会让链表变为原状。因此，新链表的最后一个节点为原链表的第 $(n - 1) - (k mod n)$ 个节点（从 $0$ 开始计数）。

这样，我们可以先将给定的链表连接成环，然后将指定位置断开。

``` c++
class Solution {
public:
    ListNode* rotateRight(ListNode* head, int k) {
        if (k == 0 || head == nullptr || head->next == nullptr) {
            return head;
        }
        int n = 1;
        ListNode* iter = head;
        while (iter->next != nullptr) {
            iter = iter->next;
            n++;
        }
        int add = n - k % n;
        if (add == n) {
            return head;
        }
        iter->next = head;
        while (add--) {
            iter = iter->next;
        }
        ListNode* ret = iter->next;
        iter->next = nullptr;
        return ret;
    }
};
```

## 不同路径

### 动态规划

用 $f(i, j)$ 表示从左上角走到 $(i, j)$ 的路径数量，其中 $i$ 和 $j$ 的范围分别是 $[0, m)$ 和 $[0, n)$。

由于我们每一步只能从向下或者向右移动一步，因此要想走到 $(i, j)$，如果向下走一步，那么会从 $(i-1, j)$ 走过来；如果向右走一步，那么会从 $(i, j-1)$ 走过来。因此我们可以写出动态规划转移方程：
$$
f(i,j)=f(i-1,j)+f(i,j-1)
$$
所有的 $f(0, j)$ 以及 $f(i, 0)$ 都设置为边界条件，它们的值均为 $1$

``` c++
class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<vector<int>> f(m, vector<int>(n));
        for (int i = 0; i < m; ++i) {
            f[i][0] = 1;
        }
        for (int j = 0; j < n; ++j) {
            f[0][j] = 1;
        }
        for (int i = 1; i < m; ++i) {
            for (int j = 1; j < n; ++j) {
                f[i][j] = f[i - 1][j] + f[i][j - 1];
            }
        }
        return f[m - 1][n - 1];
    }
};
```

### 组合数学

从左上角到右下角的过程中，我们需要移动 $m+n-2$ 次，其中有 $m-1$ 次向下移动，$n-1$ 次向右移动。因此路径的总数，就等于从 $m+n-2$ 次移动中选择 $m-1$ 次向下移动的方案数，即组合数：
$$
C_{m+n-2}^{m-1}=(\begin{matrix}m+n-2 \\ m-1 \end{matrix})=\frac{(m+n-2)(m+n-3)\cdots n}{(m-1)!}=\frac{(m+n-2)!}{(m-1)!(n-1)!}
$$

``` c++
class Solution {
public:
    int uniquePaths(int m, int n) {
        long long ans = 1;
        for (int x = n, y = 1; y < m; ++x, ++y) {
            ans = ans * x / y;
        }
        return ans;
    }
};
```

## 不同路径2

用 $f(i, j)$ 来表示从坐标 $(0, 0)$ 到坐标 $(i, j)$ 的路径总数，$u(i, j)$ 表示坐标 $(i, j)$ 是否可行，如果坐标 $(i, j)$ 有障碍物，$u(i, j) = 0$，否则 $u(i, j) = 1$。可以得到动态转移方程：
$$
f(i,j)=\begin{cases}0, & u(i,j)=0 \\ f(i-1,j) + f(i,j-1), &u(i,j) \neq0\end{cases}
$$
由于这里 $f(i, j)$ 只与 $f(i - 1, j)$ 和 $f(i, j - 1)$ 相关，我们可以运用**滚动数组思想**把空间复杂度优化称 $O(m)$

``` c++
class Solution {
public:
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
        int n = obstacleGrid.size(), m = obstacleGrid.at(0).size();
        vector <int> f(m);

        f[0] = (obstacleGrid[0][0] == 0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (obstacleGrid[i][j] == 1) {
                    f[j] = 0;
                    continue;
                }
                if (j - 1 >= 0 && obstacleGrid[i][j - 1] == 0) {
                    f[j] += f[j - 1];
                }
            }
        }

        return f.back();
    }
};
```

### 动态规划思想

当我们不熟悉的时候，怎么想到用动态规划来解决这个问题呢？我们需要从问题本身出发，寻找一些有用的信息，例如本题中：

- $(i,j)$ 位置只能从 $(i - 1, j)$ 和 $(i, j - 1)$ 走到，这样的条件就是在告诉我们这里转移是 **无后效性** 的，$f(i, j)$ 和任何的 $f(i', j')(i' > i, j' > j)$  无关；
- 动态规划的题目分为两大类，一种是求最优解类，典型问题是背包问题，另一种就是计数类，比如这里的统计方案数的问题，它们都存在一定的递推性质。前者的递推性质还有一个名字，叫做 **最优子结构** ——即当前问题的最优解取决于子问题的最优解，后者类似，当前问题的方案数取决于子问题的方案数。所以在遇到求方案数的问题时，我们可以往动态规划的方向考虑。

## 有效数字

### 确定有限状态机

#### 预备知识

确定有限状态自动机（以下简称「自动机」）是一类计算模型。它包含一系列状态，这些状态中：

- 有一个特殊的状态，被称作**初始状态**。
- 还有一系列状态被称为**接受状态**，它们组成了一个特殊的集合。其中，一个状态可能既是**初始状态**，也是**接受状态**。

起初，这个自动机处于**初始状态**。随后，它顺序地读取字符串中的每一个字符，并根据当前状态和读入的字符，按照某个事先约定好的**转移规则**，从当前状态转移到下一个状态；当状态转移完成后，它就读取下一个字符。当字符串全部读取完毕后，如果自动机处于某个**接受状态**，则判定该字符串**被接受**；否则，判定该字符串**被拒绝**。

#### 思路与算法

定义自动机的**状态集合**：

0. 初始状态

1.  符号位

2. 整数部分
3. 左侧有整数的小数点
4. 左侧无整数的小数点
5. 小数部分
6. 字符e
7. 指数部分的符号位
8. 指数部分的整数部分

下一步是找出**初始状态**和**接受状态**的集合。根据题意，**初始状态**应当为状态 0，而**接受状态**的集合则为状态 2、状态 3、状态 5 以及状态 8。换言之，字符串的末尾要么是空格，要么是数字，要么是小数点，但前提是小数点的前面有数字。

最后，需要定义**转移规则**。

<img src="https://assets.leetcode-cn.com/solution-static/65/1.png" width="1000" height="400">
在实际代码中，我们需要处理转移失败的情况。为了处理这种情况，我们可以创建一个特殊的拒绝状态。如果当前状态下没有对应读入字符的**转移规则**，我们就转移到这个特殊的拒绝状态。一旦自动机转移到这个特殊状态，我们就可以立即判定该字符串不**被接受**。

``` c++
class Solution {
public:
    enum State {
        STATE_INITIAL,
        STATE_INT_SIGN,
        STATE_INTEGER,
        STATE_POINT,
        STATE_POINT_WITHOUT_INT,
        STATE_FRACTION,
        STATE_EXP,
        STATE_EXP_SIGN,
        STATE_EXP_NUMBER,
        STATE_END,
    };

    enum CharType {
        CHAR_NUMBER,
        CHAR_EXP,
        CHAR_POINT,
        CHAR_SIGN,
        CHAR_ILLEGAL,
    };

    CharType toCharType(char ch) {
        if (ch >= '0' && ch <= '9') {
            return CHAR_NUMBER;
        } else if (ch == 'e' || ch == 'E') {
            return CHAR_EXP;
        } else if (ch == '.') {
            return CHAR_POINT;
        } else if (ch == '+' || ch == '-') {
            return CHAR_SIGN;
        } else {
            return CHAR_ILLEGAL;
        }
    }

    bool isNumber(string s) {
        unordered_map<State, unordered_map<CharType, State>> transfer{
            {
                STATE_INITIAL, {
                    {CHAR_NUMBER, STATE_INTEGER},
                    {CHAR_POINT, STATE_POINT_WITHOUT_INT},
                    {CHAR_SIGN, STATE_INT_SIGN},
                }
            }, {
                STATE_INT_SIGN, {
                    {CHAR_NUMBER, STATE_INTEGER},
                    {CHAR_POINT, STATE_POINT_WITHOUT_INT},
                }
            }, {
                STATE_INTEGER, {
                    {CHAR_NUMBER, STATE_INTEGER},
                    {CHAR_EXP, STATE_EXP},
                    {CHAR_POINT, STATE_POINT},
                }
            }, {
                STATE_POINT, {
                    {CHAR_NUMBER, STATE_FRACTION},
                    {CHAR_EXP, STATE_EXP},
                }
            }, {
                STATE_POINT_WITHOUT_INT, {
                    {CHAR_NUMBER, STATE_FRACTION},
                }
            }, {
                STATE_FRACTION,
                {
                    {CHAR_NUMBER, STATE_FRACTION},
                    {CHAR_EXP, STATE_EXP},
                }
            }, {
                STATE_EXP,
                {
                    {CHAR_NUMBER, STATE_EXP_NUMBER},
                    {CHAR_SIGN, STATE_EXP_SIGN},
                }
            }, {
                STATE_EXP_SIGN, {
                    {CHAR_NUMBER, STATE_EXP_NUMBER},
                }
            }, {
                STATE_EXP_NUMBER, {
                    {CHAR_NUMBER, STATE_EXP_NUMBER},
                }
            }
        };

        int len = s.length();
        State st = STATE_INITIAL;

        for (int i = 0; i < len; i++) {
            CharType typ = toCharType(s[i]);
            if (transfer[st].find(typ) == transfer[st].end()) {
                return false;
            } else {
                st = transfer[st][typ];
            }
        }
        return st == STATE_INTEGER || st == STATE_POINT || st == STATE_FRACTION || st == STATE_EXP_NUMBER || st == STATE_END;
    }
};
```

## 文本左右对齐

### 思路

1. 先取出一行能够容纳的单词，将这些单词根据规则填入一行

2. 计算出**额外空格**的数量 $spaceCount$，**额外空格**就是正常书写用不到的空格

   - 由总长度算起

   - 除去每个单词末尾必须的空格，为了统一处理在结尾处虚拟加上一个长度

   - 除去所有单词的长度

<img src="https://pic.leetcode-cn.com/76ce03022426fbad207b21c669fd2f68ed311f1d743bad78896dbff4aff572bd-%E5%9B%BE%E7%89%87.png" width="800" height="400">

3. 按照单词的间隙数量 $wordCount-1$，对额外空格平均分布：对于每个单词填充之后，需要填充的空格数量等于 $spaceSuffix+spaceAvg+((i-begin)<spaceExtra)$
   - $spaceSuffix$ 单词尾部固定的空格
   - $spaceAvg$ 额外空格的平均值，每个间隙都要填入
   - $(i-begin)<spaceExtra$ 额外空格的余数，前 $spaceExtra$ 个间隙需要多1个空格

<img src="https://pic.leetcode-cn.com/a423b03e2bc1a130ec2282c398a8089c0906d731db73b5d80d42190276bc7a33-%E5%9B%BE%E7%89%87.png" width="800" height="400">

4. 特殊处理
   - 一行只有一个单词，单词左对齐，右侧填满空格
   - 最后一行，所有单词左对齐，中间只有一个空格，最后一个单词右侧填满空格

``` c++
    string fillWords(vector<string>& words, int begin, int end, int maxWidth, bool lastLine = false)
    {
        int wordCount = end - begin + 1;
        int spaceCount = maxWidth + 1 - wordCount;  // 除去每个单词尾部空格， + 1 是最后一个单词的尾部空格的特殊处理
        for (int i = begin; i <= end; i++)
        {
            spaceCount -= words[i].size();  // 除去所有单词的长度
        }

        int spaceSuffix = 1;    // 词尾空格
        int spaceAvg = (wordCount == 1) ? 1 : spaceCount / (wordCount - 1);     // 额外空格的平均值
        int spaceExtra = (wordCount == 1) ? 0 : spaceCount % (wordCount - 1);   // 额外空格的余数

        string ans;
        for (int i = begin; i < end; i++)
        {
            ans += words[i];    // 填入单词
            if (lastLine)   // 特殊处理最后一行
            {
                fill_n(back_inserter(ans), 1, ' ');
                continue;
            }
            fill_n(back_inserter(ans), spaceSuffix + spaceAvg + ((i - begin) < spaceExtra), ' ');  // 根据计算结果补上空格
        }
        ans += words[end];   // 填入最后一个单词
        fill_n(back_inserter(ans), maxWidth - ans.size(), ' '); // 补上这一行最后的空格
        return ans;
    }

    vector<string> fullJustify(vector<string>& words, int maxWidth) 
    {
        vector<string> ans;
        int cnt = 0;
        int begin = 0;
        for (int i = 0; i < words.size(); i++)
        {
            cnt += words[i].size() + 1;

            if (i + 1 == words.size() || cnt + words[i + 1].size() > maxWidth)
            {   // 如果是最后一个单词，或者加上下一个词就超过长度了，即可凑成一行
                ans.push_back(fillWords(words, begin, i, maxWidth, i + 1 == words.size()));
                begin = i + 1;
                cnt = 0;
            }
        }
        return ans;
    }
```

## 平均分布

### 题目

> 在你的面前从左到右摆放着 $n$ 根长短不一的木棍，你每次可以折断一根木棍，并将折断后得到的两根木棍一左一右放在原来的位置（即若原木棍有左邻居，则两根新木棍必须放在左邻居的右边，若原木棍有右邻居，新木棍必须放在右邻居的左边，所有木棍保持左右排列）。折断后的两根木棍的长度必须为整数，且它们之和等于折断前的木棍长度。你希望最终从左到右的木棍长度 **单调不减**，那么你需要折断多少次呢？

### 思路

1. 从后往前遍历，使用 **单调递减栈**
2. 在原本应该出栈的时机，将那根木棒折断成若干小于等于栈顶的小木棒
3. 并让他们尽量保持平均大小，将略小一的入栈

<img src="https://pic.leetcode-cn.com/a7fc70d47e1b0706a62f6bda284f61360d56cdd36038ffdc5419e901cbc1f266-%E5%9B%BE%E7%89%87.png" width="680" height="400">

``` c++
int breakNum(vector<int>& nums) {
    int ans = 0;
    for (int i = nums.size() - 2; i >= 0; i--) {
        if (nums[i + 1] >= nums[i]) continue;
        int t = (nums[i] - 1) / nums[i + 1];
        ans += t;
        nums[i] /= (t + 1);
    }
    return ans;
}
```

## x的平方根

### 二分查找

由于 $x$ 平方根的整数部分 $ans$ 是满足 $k^2 \leq x$ 的最大 $k$ 值，因此我们可以对 $k$ 进行二分查找，从而得到答案。

``` c++
class Solution {
public:
    int mySqrt(int x) {
        int l = 0, r = x, ans = -1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if ((long long)mid * mid <= x) {
                ans = mid;
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
        return ans;
    }
};
```

### 牛顿迭代

牛顿迭代法是一种可以用来快速求解函数零点的方法。

为了叙述方便，我们用 $C$ 表示待求出平方根的那个整数。显然，$C$ 的平方根就是函数
$$
y=f(x)=x^2-C
$$
的零点。

牛顿迭代法的本质是借助 **泰勒级数**，从初始值开始快速向零点逼近。我们任取一个 $x_0$ 作为初始值，在每一步的迭代中，我们找到函数图像上的点 $(x_i, f(x_i))$，过该点作一条斜率为该点导数 $f'(x_i)$ 的直线，与横轴的交点记为 $x_{i+1}$。$x_{i+1}$ 相较于 $x_i$ 而言距离零点更近。在经过多次迭代后，我们就可以得到一个距离零点非常接近的交点。下图给出了从 $x_0$ 开始迭代两次，得到 $x_1$ 和 $x_2$ 的过程：

<img src="https://assets.leetcode-cn.com/solution-static/69/69_fig1.png" width="800" height="400"> 

### 算法

我们选择 $x_0=C$ 作为初始值。

在每一步迭代中，我们通过当前的交点 $x_i$，找到函数图像上的点 $(x_i,x_i^2-C)$，作一条斜率 $f'(x_i)=2x_i$ 的直线，直线的方程为：
$$
y_l = 2x_i(x-x_i)+x_i^2-C \\
	= 2x_ix-(x_i^2+C)
$$
与横轴的交点为方程 $2x_ix-(x_i^2-C)=0$ 的解，即为新的迭代结果 $x_{i+1}$：
$$
x_{i+1} = \frac{1}{2}(x_i+\frac{C}{x_i})
$$
在进行 $k$ 次迭代后，$x_k$ 的值与真实的零点 $\sqrt{C}$ 足够接近，即可作为答案。

### 细节

- 迭代到何时才算结束？
  - 每一次迭代后，我们都会距离零点更进一步，所以当相邻两次迭代得到的交点非常接近时，我们就可以断定，此时的结果已经足够我们得到答案了。一般来说，可以判断相邻两次迭代的结果的差值是否小于一个极小的非负数 $\epsilon$，其中 $\epsilon$ 一般可以取 $10^{-6}$ 或 $10^{-7}$。

- 为什么选择 $x_0=C$  作为初始值？
  - 因为 $y=f(x)=x^2-C$ 有两个零点 $-\sqrt{C}$ 和 $\sqrt{C}$。如果我们取的初始值较小，可能会迭代到 $-\sqrt{C}$ 这个零点，而我们希望找到的是 $\sqrt{C}$ 这个零点。因此选择 $x_0 = C$ 作为初始值，每次迭代均有 $x_{i+1} < x_i$，零点 $\sqrt{C}$在其左侧，所以我们一定会迭代到这个零点。

``` c++
class Solution {
public:
    int mySqrt(int x) {
        if (x == 0) {
            return 0;
        }

        double C = x, x0 = x;
        while (true) {
            double xi = 0.5 * (x0 + C / x0);
            if (fabs(x0 - xi) < 1e-7) {
                break;
            }
            x0 = xi;
        }
        return int(x0);
    }
};
```

### 袖珍计算器

用指数函数 $\exp$ 和对数函数 $\ln$ 代替平方根函数的方法：
$$
\sqrt{x}=x^{1/2}=(e^{\ln x})^{1/2}=e^{\frac{1}{2}\ln x}
$$
**注意**：由于计算机无法存储浮点数的精确值，而指数函数和对数函数的参数和返回值均为浮点数，因此运算过程中会存在误差。在得到结果的整数部分 $ans$ 后，我们应当找出 $ans$ 与 $ans+1$ 中哪一个是真正的答案。

``` c++
class Solution {
public:
    int mySqrt(int x) {
        if (x == 0) {
            return 0;
        }
        int ans = exp(0.5 * log(x));
        return ((long long)(ans + 1) * (ans + 1) <= x ? ans + 1 : ans);
    }
};
```

## 爬楼梯

### 动态规划(滚动数组)

用 $f(x)$ 表示爬到第 $x$ 级台阶的方案数，可以很容易的写出状态转移方程：
$$
f(x)=f(x-1)+f(x-2)
$$
边界条件有 $f(0)=1，f(1)=1$。

我们不难通过转移方程和边界条件给出一个时间复杂度和空间复杂度都是 $O(n)$ 的实现，但是由于这里的 $f(x)$只和 $f(x - 1)$ 与 $f(x - 2)$ 有关，所以我们可以用**滚动数组思想**把空间复杂度优化成 $O(1)$。

<img src="https://assets.leetcode-cn.com/solution-static/70/70_fig1.gif" width="800" height="360">

``` c++
class Solution {
public:
    int climbStairs(int n) {
        int p = 0, q = 0, r = 1;
        for (int i = 1; i <= n; ++i) {
            p = q; 
            q = r; 
            r = p + q;
        }
        return r;
    }
};
```

### 快速矩阵幂

以上的方法适用于 $n$ 比较小的情况，在 $n$ 变大之后，$O(n)$ 的时间复杂度会让这个算法看起来有些捉襟见肘。我们可以用**矩阵快速幂**的方法来优化这个过程。

我们构建如下递进关系：
$$
\left[ \begin{matrix}1 & 1 \\ 1 & 0\end{matrix} \right] \left[ \begin{matrix} f(n) \\ f(n-1) \end{matrix} \right]= 
\left[ \begin{matrix} f(n)+f(n-1) \\ f(n) \end{matrix} \right] = \left[ \begin{matrix} f(n+1) \\ f(n) \end{matrix} \right]
$$
因此：
$$
\left[ \begin{matrix} f(n+1) \\ f(n) \end{matrix} \right] =\left[ \begin{matrix}1 & 1 \\ 1 & 0\end{matrix} \right]^n
\left[ \begin{matrix} f(1) \\ f(0) \end{matrix} \right]
$$
令：
$$
M =\left[ \begin{matrix}1 & 1 \\ 1 & 0\end{matrix} \right]
$$
我们只要能快速计算矩阵 $M$ 的 $n$ 次幂，就可以得到 $f(n)$ 的值。

``` c++
class Solution {
public:
    vector<vector<long long>> multiply(vector<vector<long long>> &a, vector<vector<long long>> &b) {
        vector<vector<long long>> c(2, vector<long long>(2));
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j];
            }
        }
        return c;
    }

    vector<vector<long long>> matrixPow(vector<vector<long long>> a, int n) {
        vector<vector<long long>> ret = {{1, 0}, {0, 1}};
        while (n > 0) {
            if ((n & 1) == 1) {
                ret = multiply(ret, a);
            }
            n >>= 1;
            a = multiply(a, a);
        }
        return ret;
    }

    int climbStairs(int n) {
        vector<vector<long long>> ret = {{1, 1}, {1, 0}};
        vector<vector<long long>> res = matrixPow(ret, n);
        return res[0][0];
    }
};
```

#### 如何想到用矩阵快速幂

- 如果一个问题可与转化为求解一个矩阵的 $n$ 次方的形式，那么可以用快速幂来加速计算
- 如果一个递推式形如 $f(n)=\sum_{i=1}^ma_if(n-1)$，即齐次线性递推式，我们就可以把数列的递推关系转化为矩阵的递推关系，即构造出一个矩阵的 $n$ 次方乘以一个列向量得到一个列向量，这个列向量中包含我们要求的 $f(n)$：

$$
\left[ \begin{matrix} a_1 & a_2 & a_3 & \cdots & a_n \\ 1 & 0 & 0 & \cdots & 0 \\
					  0 & 1 & 0 & \cdots & 0 \\ 0 & 0 & 1 & \cdots & 0 \\
					  \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \cdots & 1\end{matrix} \right]
$$

### 通项公式

由：
$$
M =\left[ \begin{matrix}1 & 1 \\ 1 & 0\end{matrix} \right]
$$
求出斐波那切数列的特征方程：
$$
det(M-\lambda I) = \left| \begin{matrix} 1-\lambda & 1 \\ 1 & -\lambda \end{matrix} \right|=\lambda^2-\lambda-1
$$
令 $det(M-\lambda I) = 0$，得到：
$$
\lambda^2 = \lambda+1
$$
求得：
$$
\lambda_1 = \frac{1+\sqrt{5}}{2}，\lambda_2=\frac{1-\sqrt{5}}{2}
$$
设通解为$f(n)=c_1x_1^2+c_2x_2^n$，代入初始条件 $f(1)=1, f(2)=1$，得 $c_1=\frac{1}{\sqrt{5}}$，$c_2=-\frac{1}{\sqrt{5}}$，所以递推数列的通项公式为：
$$
f(n)=\frac{1}{\sqrt{5}} \left[ (\frac{1+\sqrt{5}}{2})^2 - (\frac{1-\sqrt{5}}{2})^2 \right]
$$

``` c++
class Solution {
public:
    int climbStairs(int n) {
        double sqrt5 = sqrt(5);
        double fibn = pow((1 + sqrt5) / 2, n + 1) - pow((1 - sqrt5) / 2, n + 1);
        return (int)round(fibn / sqrt5);
    }
};
```

## 简化路径

Linux的目录层级是用**栈**实现的。

- 分割字符串之后根据每种情况进行判定
- $.$ 和 $''$ 不用管，直接跳过
- $..$ 代表返回上一级，弹出栈顶元素（注意判空）
- 其它情况直接入栈

``` c++
class Solution {
public:
    string simplifyPath(string path) {
        stringstream is(path);
        vector<string> strs;
        string res = "", tmp = "";
        while(getline(is, tmp, '/')) {
            if(tmp == "" || tmp == ".")
                continue;
            else if(tmp == ".." && !strs.empty())
                strs.pop_back();
            else if(tmp != "..")
                strs.push_back(tmp);
        }
        for(string str:strs) 
            res +=  "/" + str;
        if(res.empty())
            return "/";
        return res;
    }
};
```

自己写的不用字符串流处理的垃圾代码

``` c++
class Solution {
public:
    string simplifyPath(string path) {
        int pos = 0, nextPos = 0;
        vector<string> strs;
        string ans;
        while((nextPos = path.find_first_of('/', pos+1)) != string::npos) {
            string str = path.substr(pos+1, nextPos-pos-1);
            if (str == "" || str == ".") {
                pos = nextPos;
                continue;
            } 
            else if (str == "..") {
                if (!strs.empty())
                    strs.pop_back();
            }
            else
                strs.push_back(str);
            pos = nextPos;
        }

        if (pos != path.size()-1) {
            string str = path.substr(pos+1, nextPos-pos-1);
            if (str == "" || str == ".")
                ;
            else if (str == "..") {
                if (!strs.empty())
                    strs.pop_back();
            }
            else
                strs.push_back(str);
        }
        

        for (auto &s: strs)
            ans += "/" + s;

        if (ans.empty())
            return "/";

        return ans;
    }
};
```

## 编辑距离

### 动态规划

我们可以发现，如果我们有两个单词 `A` 和单词 `B`：

- 对单词 `A` 删除一个字符和对单词 `B` 插入一个字符是等价的。例如当单词 `A` 为 `doge`，单词 `B` 为 `dog` 时，我们既可以删除单词 `A` 的最后一个字符 `e`，得到相同的 `dog`，也可以在单词 `B` 末尾添加一个字符 `e`，得到相同的 `doge`；
- 同理，对单词 `B` 删除一个字符和对单词 `A` 插入一个字符也是等价的；
- 对单词 `A` 替换一个字符和对单词 `B` 替换一个字符是等价的。例如当单词 `A` 为 `bat`，单词 `B` 为 `cat` 时，我们修改单词 `A` 的第一个字母 `b->c`，和修改单词 B 的第一个字母 `c -> b` 是等价的。

因此，本质不同的操作实际上只有三种：

- 在单词 `A` 中插入一个字符；
- 在单词 `B` 中插入一个字符；
- 修改单词 `A` 的一个字符。

这样以来，我们就可以把原问题转化为规模较小的子问题。我们用 `A = horse`，`B = ros` 作为例子，来看一看是如何把这个问题转化为规模较小的若干子问题的。

- **在单词 A 中插入一个字符**：如果我们知道 `horse` 到 `ro` 的编辑距离为 `a`，那么显然 `horse` 到 `ros` 的编辑距离不会超过 `a + 1`。这是因为我们可以在 `a` 次操作后将 `horse` 和 `ro` 变为相同的字符串，只需要额外的 `1` 次操作，在单词 `A` 的末尾添加字符 `s`，就能在 `a + 1` 次操作后将 `horse` 和 `ro` 变为相同的字符串；
- **在单词 B 中插入一个字符**：如果我们知道 `hors` 到 `ros` 的编辑距离为 `b`，那么显然 `horse` 到 `ros` 的编辑距离不会超过 `b + 1`，原因同上；
- **修改单词 A 的一个字符**：如果我们知道 `hors` 到 `ro` 的编辑距离为 `c`，那么显然 `horse` 到 `ros` 的编辑距离不会超过 `c + 1`，原因同上。

那么从 `horse` 变成 `ros` 的编辑距离应该为 `min(a + 1, b + 1, c + 1)`。

**注意**：为什么我们总是在单词 `A` 和 `B` 的末尾插入或者修改字符，能不能在其它的地方进行操作呢？答案是可以的，但是我们知道，操作的顺序是不影响最终的结果的。例如对于单词 `cat`，我们希望在 `c` 和 `a` 之间添加字符 `d` 并且将字符 `t` 修改为字符 `b`，那么这两个操作无论为什么顺序，都会得到最终的结果 `cdab`。

我们可以继续用上面的方法拆分这个问题，对于这个问题拆分出来的所有子问题，我们也可以继续拆分，直到：

- 字符串 `A` 为空，如从 ` ` 转换到 `ro`，显然编辑距离为字符串 `B` 的长度，这里是 `2`；
- 字符串 `B` 为空，如从 `horse` 转换到 ` `，显然编辑距离为字符串 `A` 的长度，这里是 `5`。

因此，我们就可以使用**动态规划**来解决这个问题了。我们用 $D[i][j]$ 表示 `A` 的前 $i$ 个字母和 `B` 的前 $j$ 个字母之间的编辑距离。

如上所述，当我们获得 $D[i][j-1]$，$D[i-1][j]$ 和 $D[i-1][j-1]$ 的值之后就可以计算出 $D[i][j]$。

- $D[i][j-1]$ 为 `A` 的前 $i$ 个字符和 `B` 的前 $j - 1$ 个字符编辑距离的子问题。即对于 `B` 的第 $j$ 个字符，我们在 `A` 的末尾添加了一个相同的字符，那么 $D[i][j]$ 最小可以为 $D[i][j-1] + 1$；
- $D[i-1][j]$ 为 `A` 的前 $i - 1$ 个字符和 `B` 的前 $j$ 个字符编辑距离的子问题。即对于 `A` 的第 $i$ 个字符，我们在 `B` 的末尾添加了一个相同的字符，那么 $D[i][j]$ 最小可以为 $D[i-1][j] + 1$;

- $D[i-1][j-1]$ 为 `A` 前 $i - 1$ 个字符和 `B` 的前 $j - 1$ 个字符编辑距离的子问题。即对于 `B` 的第 $j$ 个字符，我们修改 `A` 的第 $i$ 个字符使它们相同，那么 $D[i][j]$ 最小可以为 $D[i-1][j-1] + 1$。特别地，如果 `A` 的第 $i$ 个字符和 `B` 的第 $j$ 个字符原本就相同，那么我们实际上不需要进行修改操作。在这种情况下，$D[i][j]$ 最小可以为 $D[i-1][j-1]$。

那么我们可以写出如下的状态转移方程：

- 若 `A` 和 `B` 的最后一个字母相同：

$$
D[i][j] = min(D[i][j−1]+1,D[i−1][j]+1,D[i−1][j−1])\\
=1+min(D[i][j−1],D[i−1][j],D[i−1][j−1]−1)
$$

- 若 `A` 和 `B` 的最后一个字母不同：

$$
D[i][j]=1+min(D[i][j−1],D[i−1][j],D[i−1][j−1])
$$

示意如下：

<img src="https://pic.leetcode-cn.com/3241789f2634b72b917d769a92d4f6e38c341833247391fb1b45eb0441fe5cd2-72_fig2.PNG" width="700" height="400">

对于边界情况，一个空串和一个非空串的编辑距离为 $D[i][0] = i$ 和 $D[0][j] = j$，$D[i][0]$ 相当于对 `word1` 执行 $i$ 次删除操作，$D[0][j]$ 相当于对 `word1` 执行 $j$ 次插入操作。

``` c++
class Solution {
public:
    int minDistance(string word1, string word2) {
        int n = word1.length();
        int m = word2.length();

        // 有一个字符串为空串
        if (n * m == 0) return n + m;

        // DP 数组
        vector<vector<int>> D(n+1, vector<int>(m+1));

        // 边界状态初始化
        for (int i = 0; i < n + 1; i++) {
            D[i][0] = i;
        }
        for (int j = 0; j < m + 1; j++) {
            D[0][j] = j;
        }

        // 计算所有 DP 值
        for (int i = 1; i < n + 1; i++) {
            for (int j = 1; j < m + 1; j++) {
                int left = D[i - 1][j] + 1;
                int down = D[i][j - 1] + 1;
                int left_down = D[i - 1][j - 1];
                if (word1[i - 1] != word2[j - 1]) left_down += 1;
                D[i][j] = min(left, min(down, left_down));

            }
        }
        return D[n][m];
    }
};
```

## 矩阵置零

### 使用标记数组

用两个标记数组分别记录每一行和每一列是否有零出现。

``` c++
class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) {
        int m = matrix.size();
        int n = matrix[0].size();
        vector<int> row(m), col(n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (!matrix[i][j]) {
                    row[i] = col[j] = true;
                }
            }
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (row[i] || col[j]) {
                    matrix[i][j] = 0;
                }
            }
        }
    }
};
```

- 时间复杂度：$O(mn)$
- 空间复杂度：$O(m+n)$

### 使用两个标记变量

可以用矩阵的第一行和第一列代替方法一中的两个标记数组，以达到` O(1)` 的额外空间。但这样会导致原数组的第一行和第一列被修改，无法记录它们是否原本包含 `0`。因此我们需要额外使用两个标记变量分别记录第一行和第一列是否原本包含 `0`。

``` c++
class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) {
        int m = matrix.size();
        int n = matrix[0].size();
        int flag_col0 = false, flag_row0 = false;
        for (int i = 0; i < m; i++) {
            if (!matrix[i][0]) {
                flag_col0 = true;
            }
        }
        for (int j = 0; j < n; j++) {
            if (!matrix[0][j]) {
                flag_row0 = true;
            }
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (!matrix[i][j]) {
                    matrix[i][0] = matrix[0][j] = 0;
                }
            }
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (!matrix[i][0] || !matrix[0][j]) {
                    matrix[i][j] = 0;
                }
            }
        }
        if (flag_col0) {
            for (int i = 0; i < m; i++) {
                matrix[i][0] = 0;
            }
        }
        if (flag_row0) {
            for (int j = 0; j < n; j++) {
                matrix[0][j] = 0;
            }
        }
    }
};
```

- 时间复杂度：$O(mn)$
- 空间复杂度：$O(1)$

### 使用一个标记变量

我们可以对方法二进一步优化，只使用一个标记变量记录第一列是否原本存在 $0$。这样，第一列的第一个元素即可以标记第一行是否出现 $0$。但为了防止每一列的第一个元素被提前更新，我们需要从最后一行开始，倒序地处理矩阵元素。

``` c++
class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) {
        int m = matrix.size();
        int n = matrix[0].size();
        int flag_col0 = false;
        for (int i = 0; i < m; i++) {
            if (!matrix[i][0]) {
                flag_col0 = true;
            }
            for (int j = 1; j < n; j++) {
                if (!matrix[i][j]) {
                    matrix[i][0] = matrix[0][j] = 0;
                }
            }
        }
        for (int i = m - 1; i >= 0; i--) {
            for (int j = 1; j < n; j++) {
                if (!matrix[i][0] || !matrix[0][j]) {
                    matrix[i][j] = 0;
                }
            }
            if (flag_col0) {
                matrix[i][0] = 0;
            }
        }
    }
};
```

- 时间复杂度：$O(mn)$
- 空间复杂度：$O(1)$

## 搜索二维矩阵

### 两次二分查找

我们可以对矩阵的第一列的元素二分查找，找到最后一个不大于目标值的元素，然后在该元素所在行中二分查找目标值是否存在。

``` c++
class Solution {
public:
    bool searchMatrix(vector<vector<int>> matrix, int target) {
        auto row = upper_bound(matrix.begin(), matrix.end(), target, [](const int b, const vector<int> &a) {
            return b < a[0];
        });
        if (row == matrix.begin()) {
            return false;
        }
        --row;
        return binary_search(row->begin(), row->end(), target);
    }
};
```

### 一次二分查找

若将矩阵每一行拼接在上一行的末尾，则会得到一个升序数组，我们可以在该数组上二分找到目标元素。

代码实现时，可以二分升序数组的下标，将其映射到原矩阵的行和列上。

``` c++
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int m = matrix.size(), n = matrix[0].size();
        int low = 0, high = m * n - 1;
        while (low <= high) {
            int mid = (high - low) / 2 + low;
            int x = matrix[mid / n][mid % n];
            if (x < target) {
                low = mid + 1;
            } else if (x > target) {
                high = mid - 1;
            } else {
                return true;
            }
        }
        return false;
    }
};
```

## 颜色分类

### 单指针

对数组进行两次遍历。在第一次遍历中，我们将数组中所有的 $0$ 交换到数组的头部。在第二次遍历中，我们将数组中所有的 $1$ 交换到头部的 $0$ 之后。此时，所有的 $2$ 都出现在数组的尾部，这样我们就完成了排序。

``` c++
class Solution {
public:
    void sortColors(vector<int>& nums) {
        int n = nums.size();
        int ptr = 0;
        for (int i = 0; i < n; ++i) {
            if (nums[i] == 0) {
                swap(nums[i], nums[ptr]);
                ++ptr;
            }
        }
        for (int i = ptr; i < n; ++i) {
            if (nums[i] == 1) {
                swap(nums[i], nums[ptr]);
                ++ptr;
            }
        }
    }
};
```

### 双指针

我们用指针 $p_0$ 来交换 $0$，$p_1$ 来交换 $1$，初始值都为 $0$。当我们从左向右遍历整个数组时：

- 如果找到了 $1$，那么将其与 $nums[p1]$ 进行交换，并将 $p_1$ 向后移动一个位置；
- 如果找到了 $0$，因为连续的 $0$ 之后是连续的 $1$，因此如果我们将 $0$ 与 $nums[p_0]$ 进行交换，那么我们可能会把一个 $1$ 交换出去。当 $p_0 < p_1$ 时，我们已经将一些 $1$ 连续地放在头部，此时一定会把一个 $1$ 交换出去，导致答案错误。因此，如果 $p_0 < p_1$，那么我们需要再将 $nums[i]$ 与$nums[p_1]$ 进行交换，其中 $i$ 是当前遍历到的位置，在进行了第一次交换后，$nums[i]$ 的值为 $1$，我们需要将这个 $1$ 放到**头部**的末端。在最后，无论是否有 $p_0 < p_1$，我们需要将 $p_0$ 和 $p_1$ 均向后移动一个位置

``` c++
class Solution {
public:
    void sortColors(vector<int>& nums) {
        int n = nums.size();
        int p0 = 0, p1 = 0;
        for (int i = 0; i < n; ++i) {
            if (nums[i] == 1) {
                swap(nums[i], nums[p1]);
                ++p1;
            } else if (nums[i] == 0) {
                swap(nums[i], nums[p0]);
                if (p0 < p1) {
                    swap(nums[i], nums[p1]);
                }
                ++p0;
                ++p1;
            }
        }
    }
};
```

### 双指针2

我们也可以考虑使用指针 $p_0$ 来交换 $0$，$p_2$ 来交换 $2$。此时，$p_0$ 的初始值仍然为 $0$，而 $p_2$ 的初始值为 $n-1$。在遍历的过程中，我们需要找出所有的 $0$ 交换至数组的头部，并且找出所有的 $2$ 交换至数组的尾部。

由于此时其中一个指针 $p_2$ 是从右向左移动的，因此当我们在从左向右遍历整个数组时，如果遍历到的位置超过了 $p_2$，那么就可以直接停止遍历了。

具体地，我们从左向右遍历整个数组，设当前遍历到的位置为 $i$，对应的元素为 $nums[i]$；

- 如果找到了 $0$，那么与前面两种方法类似，将其与 $nums[p_0]$ 进行交换，并将 $p_0$ 向后移动一个位置；
- 如果找到了 $2$，我们需要不断地将其与 $nums[p_2]$ 进行交换，直到新的 $nums[i]$ 不为 $2$。此时，如果 $nums[i]$ 为 $0$，那么对应着第一种情况；如果 $nums[i]$ 为 $1$，那么就不需要进行任何后续的操作。这是因为：当我们将 $nums[i]$ 与 $nums[p_2]$ 进行交换之后，新的 $nums[i]$ 可能仍然是 $2$，也可能是 $0$。然而此时我们已经结束了交换，开始遍历下一个元素 $nums[i+1]$，不会再考虑 $nums[i]$ 了，这样我们就会得到错误的答案。

``` c++
class Solution {
public:
    void sortColors(vector<int>& nums) {
        int n = nums.size();
        int p0 = 0, p2 = n - 1;
        for (int i = 0; i <= p2; ++i) {
            while (i <= p2 && nums[i] == 2) {
                swap(nums[i], nums[p2]);
                --p2;
            }
            if (nums[i] == 0) {
                swap(nums[i], nums[p0]);
                ++p0;
            }
        }
    }
};
```

## 最小覆盖子串

### 滑动窗口

在滑动窗口类型的问题中都会有两个指针，一个用于**延伸**现有窗口的 $r$ 指针，和一个用于**收缩**窗口的 $l$ 指针。在任意时刻，只有一个指针运动，而另一个保持静止。**我们在 $s$ 上滑动窗口，通过移动 $r$ 指针不断扩张窗口。当窗口包含 $t$ 全部所需的字符后，如果能收缩，我们就收缩窗口直到得到最小窗口**。

<img src="https://assets.leetcode-cn.com/solution-static/76/76_fig1.gif" width="850" height="400" >

如何判断当前的窗口包含所有 $t$ 所需的字符呢？我们可以用一个哈希表表示 $t$ 中所有的字符以及它们的个数，用一个哈希表动态维护窗口中所有的字符以及它们的个数，如果这个动态表中包含 $t$ 的哈希表中的所有字符，并且对应的个数都不小于 $t$ 的哈希表中各个字符的个数，那么当前的窗口是**可行**的。

``` c++
class Solution {
public:
    unordered_map <char, int> ori, cnt;

    bool check() {
        for (const auto &p: ori) {
            if (cnt[p.first] < p.second) {
                return false;
            }
        }
        return true;
    }

    string minWindow(string s, string t) {
        for (const auto &c: t) {
            ++ori[c];
        }

        int l = 0, r = -1;
        int len = INT_MAX, ansL = -1, ansR = -1;

        while (r < int(s.size())) {
            if (ori.find(s[++r]) != ori.end()) {
                ++cnt[s[r]];
            }
            while (check() && l <= r) {
                if (r - l + 1 < len) {
                    len = r - l + 1;
                    ansL = l;
                }
                if (ori.find(s[l]) != ori.end()) {
                    --cnt[s[l]];
                }
                ++l;
            }
        }

        return ansL == -1 ? string() : s.substr(ansL, len);
    }
};
```

### 考虑优化

如果 $s = {\rm XX \cdots XABCXXXX}$，$t = {\rm ABC}$，那么显然 ${\rm [XX \cdots XABC]}$是第一个得到的「可行」区间，得到这个可行区间后，我们按照**收缩**窗口的原则更新左边界，得到最小区间。我们其实做了一些无用的操作，就是更新右边界的时候**延伸**进了很多无用的 $\rm XX$，更新左边界的时候「收缩」扔掉了这些无用的 $\rm XX$，做了这么多无用的操作，只是为了得到短短的 $\rm ABC$。其实在 $s$ 中，有的字符我们是不关心的，我们只关心 $t$ 中出现的字符，我们可不可以先预处理 $s$，扔掉那些 $t$ 中没有出现的字符，然后再做滑动窗口呢？

## 组合

### 递归(DFS)

``` c++
class Solution {
public:
    vector<int> temp;
    vector<vector<int>> ans;

    void dfs(int cur, int n, int k) {
        // 剪枝：temp 长度加上区间 [cur, n] 的长度小于 k，不可能构造出长度为 k 的 temp
        if (temp.size() + (n - cur + 1) < k) {
            return;
        }
        // 记录合法的答案
        if (temp.size() == k) {
            ans.push_back(temp);
            return;
        }
        // 考虑选择当前位置
        temp.push_back(cur);
        dfs(cur + 1, n, k);
        temp.pop_back();
        // 考虑不选择当前位置
        dfs(cur + 1, n, k);
    }

    vector<vector<int>> combine(int n, int k) {
        dfs(1, n, k);
        return ans;
    }
};
```

### 字典序法

我们把原序列中被选中的位置记为 $1$，不被选中的位置记为 $0$，对于每个方案都可以构造出一个二进制数。我们让原序列从大到小排列（即 $\{ n, n - 1, \cdots 1, 0 \}$）。以 $n=4, k=2$  为例：

| 原序列中被选中的数 | 对应的二进制数 | 方案 |
| ------------------ | -------------- | ---- |
| $43[2][1]$         | 0011           | 2,1  |
| $4[3]2[1]$         | 0101           | 3,1  |
| $4[3][2]1$         | 0110           | 3,2  |
| [4]32[1]           | 1001           | 4,1  |
| $[4]3[2]1$         | 1010           | 4,2  |
| $[4][3]21$         | 1100           | 4,3  |

我们可以看出「对应的二进制数」一列包含了由 $k$ 个 $1$ 和 $n - k$ 个 $0$ 组成的所有二进制数，并且按照字典序排列。

考虑一个二进制数数字 $x$，它由 $k$ 个 $1$ 和 $n - k$ 个 $0$ 组成，如何找到它的字典序中的下一个数字 ${\rm next}(x)$，这里分两种情况：

- 规则一：$x$ 的最低位为 $1$，这种情况下，如果末尾由 $t$ 个连续的 $1$，我们直接将倒数第 $t$ 位的 $1$ 和倒数第 $t + 1$ 位的 $0$ 替换，就可以得到 ${\rm next}(x)$。如$0011\rightarrow 0101，0101 \rightarrow 0110，1001 \rightarrow 1010，1001111 \rightarrow 1010111$
- 规则二：$x$ 的最低位为 $0$，这种情况下，末尾有 $t$ 个连续的 $0$，而这 $t$ 个连续的 $0$ 之前有 $m$ 个连续的 $1$，我们可以将倒数第 $t + m$ 位置的 $1$ 和倒数第 $t + m + 1$ 位的 $0$ 对换，然后把倒数第 $t + 1$ 位到倒数第 $t + m - 1$ 位的 $1$ 移动到最低位。如 $0110 \rightarrow 1001，1010 \rightarrow 1100，1011100 \rightarrow 1100011$

至此，我们可以写出一个朴素的程序，用一个长度为 $n$ 的 $0/1$ 数组来表示选择方案对应的二进制数，初始状态下最低的 $k$ 位全部为 $1$，其余位置全部为 $0$，然后不断通过上述方案求 $\rm next$，就可以构造出所有的方案。

我们可以进一步地优化。在朴素的方法中我们通过二进制数来构造方案，而二进制数是需要通过迭代的方法来获取 $\rm next$ 的。考虑不通过二进制数，直接在方案上变换来得到下一个方案。假设一个方案从低到高的 $k$ 个数分别是 $\{a_0, a_1, \cdots, a_{k-1}\}$，我们可以从低位向高位找到第一个 $j$ 使得 $a_{j} + 1 \neq a_{j + 1}$，我们知道出现在 $a$ 序列中的数字在二进制数中对应的位置一定是 $1$，即表示被选中，那么 $a_{j} + 1 \neq a_{j + 1}$ 意味着 $a_j$ 和 $a_{j+1}$ 对应的二进制中间有 $0$，即这两个 $1$ 不连续。我们把 $a_j$ 对应的 $1$ 向高位推送，也就意味着 $a_j \leftarrow a_{j+1}$，而对于 $i \in [0, j-1]$ 内的所有 $a_i$ 把值恢复成 $i+1$，即对应这 $j$ 个 $1$ 被移到了二进制数的最低 $j$ 位。这似乎只考虑了上面的「规则二」。但是实际上**「规则一」是「规则二」在 $t = 0$ 时的特殊情况**，因此这么做和按照两条规则模拟是等价的。

在实现的时候，我们可以用一个数组 $\rm temp$ 来存放 $a$ 序列，一开始我们先把 $1$ 到 $k$ 按顺序存入这个数组，他们对应的下标是 $0$ 到 $k - 1$。为了计算的方便，我们需要在下标 $k$ 的位置放置一个哨兵 $n + 1$。每次变换的时候，我们把第一个$a_{j} + 1 \neq a_{j + 1}$ 的 $j$ 找出，使 $a_j$ 自增 $1$，同时对 $i \in [0, j-1]$ 的 $a_i$ 重新置数。如此循环，直到 $\rm temp$ 中的所有元素为 $n$ 内最大的 $k$ 个元素。

如果 $j = k$ 了，就说明 $[0, k - 1]$ 内的所有的数字是比第 $k$ 位小的最后 $k$ 个数字，这个时候我们找不到任何方案的字典序比当前方案大了，结束枚举。

``` c++
class Solution {
public:
        vector<int> temp;
        vector<vector<int>> ans;

    vector<vector<int>> combine(int n, int k) {
        // 初始化
        // 将 temp 中 [0, k - 1] 每个位置 i 设置为 i + 1，即 [0, k - 1] 存 [1, k]
        // 末尾加一位 n + 1 作为哨兵
        for (int i = 1; i <= k; ++i) {
            temp.push_back(i);
        }
        temp.push_back(n + 1);
        
        int j = 0;
        while (j < k) {
            ans.emplace_back(temp.begin(), temp.begin() + k);
            j = 0;
            // 寻找第一个 temp[j] + 1 != temp[j + 1] 的位置 t
            // 我们需要把 [0, t - 1] 区间内的每个位置重置成 [1, t]
            while (j < k && temp[j] + 1 == temp[j + 1]) {
                temp[j] = j + 1;
                ++j;
            }
            // j 是第一个 temp[j] + 1 != temp[j + 1] 的位置
            ++temp[j];
        }
        return ans;
    }
};
```

## 子集

### 二进制

记原序列中元素的总数为 $n$。原序列中的每个数字 $a_i$ 的状态可能有两种，即**在子集中**和**不在子集中**。我们用 $1$ 表示**在子集中**，$0$ 表示**不在子集中**，那么每一个子集可以对应一个长度为 $n$ 的 $0/1$ 序列，第 $i$ 位表示 $a_i$ 是否在子集中。例如，当 $n=3$, $a=\{5,2,9\}$ 时：

| $0/1$ 序列 | 子集    | $0/1$ 序列对应的二进制数 |
| ---------- | ------- | ------------------------ |
| $000$      | {}      | 0                        |
| $001$      | {9}     | 1                        |
| $010$      | {2}     | 2                        |
| $011$      | {2,9}   | 3                        |
| $100$      | {5}     | 4                        |
| $101$      | {5,9}   | 5                        |
| $110$      | {5,2}   | 6                        |
| $111$      | {5,2,9} | 7                        |

我们可以枚举 $\textit{mask} \in [0, 2^n - 1]$，$\textit{mask}$ 的二进制表示是一个 $0/1$ 序列，我们可以按照这个 $0/1$ 序列在原集合当中取数。当我们枚举完所有 $2^n$ 个 $\textit{mask}$，我们也就能构造出所有的子集。

``` c++
class Solution {
public:
    vector<int> t;
    vector<vector<int>> ans;

    vector<vector<int>> subsets(vector<int>& nums) {
        int n = nums.size();
        for (int mask = 0; mask < (1 << n); ++mask) {
            t.clear();
            for (int i = 0; i < n; ++i) {
                if (mask & (1 << i)) {
                    t.push_back(nums[i]);
                }
            }
            ans.push_back(t);
        }
        return ans;
    }
};
```

## 单词搜索

### 深度优先搜索(DFS)

设函数 $\text{check}(i, j, k)$ 判断以网格的 $(i, j)$ 位置出发，能否搜索到单词 $\textit{word}[k..]$，其中 $\textit{word}[k..]$ 表示字符串 $\textit{word}$ 从第 $k$ 个字符开始的后缀子串。如果能搜索到，则返回 $\texttt{true}$，反之返回 $\texttt{false}$。函数 $\text{check}(i, j, k)$ 的执行步骤如下：

- 如果 $\textit{board}[i][j] \neq s[k]$，当前字符不匹配，直接返回 $\texttt{false}$。
- 如果当前已经访问到字符串的末尾，且对应字符依然匹配，此时直接返回 $\texttt{true}$。
- 否则，遍历当前位置的所有相邻位置。如果从某个相邻位置出发，能够搜索到子串 $\textit{word}[k+1..]$，则返回 $\texttt{true}$，否则返回 $\texttt{false}$。

这样，我们对每一个位置 $(i,j)$ 都调用函数 $\text{check}(i, j, 0)$ 进行检查：只要有一处返回 $\texttt{true}$，就说明网格中能够找到相应的单词，否则说明不能找到。

为了防止重复遍历相同的位置，需要额外维护一个与 $\textit{board}$ 等大的 $\textit{visited}$ 数组，用于标识每个位置是否被访问过。每次遍历相邻位置时，需要跳过已经被访问的位置。

``` c++
class Solution {
public:
    bool check(vector<vector<char>>& board, vector<vector<int>>& visited, int i, int j, string& s, int k) {
        if (board[i][j] != s[k]) {
            return false;
        } else if (k == s.length() - 1) {
            return true;
        }
        visited[i][j] = true;
        vector<pair<int, int>> directions{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        bool result = false;
        for (const auto& dir: directions) {
            int newi = i + dir.first, newj = j + dir.second;
            if (newi >= 0 && newi < board.size() && newj >= 0 && newj < board[0].size()) {
                if (!visited[newi][newj]) {
                    bool flag = check(board, visited, newi, newj, s, k + 1);
                    if (flag) {
                        result = true;
                        break;
                    }
                }
            }
        }
        visited[i][j] = false;
        return result;
    }

    bool exist(vector<vector<char>>& board, string word) {
        int h = board.size(), w = board[0].size();
        vector<vector<int>> visited(h, vector<int>(w));
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                bool flag = check(board, visited, i, j, word, 0);
                if (flag) {
                    return true;
                }
            }
        }
        return false;
    }
};
```

## 删除排序数组中的重复项2

### 双指针

因为给定数组是有序的，所以相同元素必然连续。我们可以使用双指针解决本题，遍历数组检查每一个元素是否应该被保留，如果应该被保留，就将其移动到指定位置。具体地，我们定义两个指针 $\textit{slow}$ 和 $\textit{fast}$ 分别为慢指针和快指针，其中慢指针表示处理出的数组的长度，快指针表示已经检查过的数组的长度，即 $\textit{nums}[\textit{fast}]$ 表示待检查的第一个元素，$\textit{nums}[\textit{slow} - 1]$ 为上一个应该被保留的元素所移动到的指定位置。

因为本题要求相同元素最多出现两次而非一次，所以我们需要检查上上个应该被保留的元素 $\textit{nums}[\textit{slow} - 2]$ 是否和当前待检查元素 $\textit{nums}[\textit{fast}]$ 相同。当且仅当 $\textit{nums}[\textit{slow} - 2] = \textit{nums}[\textit{fast}]$ 时，当前待检查元素 $\textit{nums}[\textit{fast}]$ 不应该被保留（因为此时必然有 $\textit{nums}[\textit{slow} - 2] = nums[\textit{slow} - 1] = \textit{nums}[\textit{fast}]$）。最后，$\textit{slow}$ 即为处理好的数组的长度。

特别地，数组的前两个数必然可以被保留，因此对于长度不超过 $2$ 的数组，我们无需进行任何处理，对于长度超过 $2$ 的数组，我们直接将双指针的初始值设为 $2$ 即可。

``` c++
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        int n = nums.size();
        if (n <= 2) {
            return n;
        }
        int slow = 2, fast = 2;
        while (fast < n) {
            if (nums[slow - 2] != nums[fast]) {
                nums[slow] = nums[fast];
                ++slow;
            }
            ++fast;
        }
        return slow;
    }
};
```

## 搜索旋转排序数组2

### 二分查找

对于数组中有重复元素的情况，二分查找时可能会有 $a[l]=a[\textit{mid}]=a[r]$，此时无法判断区间 $[l,\textit{mid}]$ 和区间 $[\textit{mid}+1,r]$ 哪个是有序的。

例如 $\textit{nums}=[3,1,2,3,3,3,3]$，$\textit{target}=2$，首次二分时无法判断区间 $[0,3]$ 和区间 $[4,6]$ 哪个是有序的。

对于这种情况，我们只能将当前二分区间的左边界加一，右边界减一，然后在新区间上继续二分查找。

``` c++
class Solution {
public:
    bool search(vector<int> &nums, int target) {
        int n = nums.size();
        if (n == 0) {
            return false;
        }
        if (n == 1) {
            return nums[0] == target;
        }
        int l = 0, r = n - 1;
        while (l <= r) {
            int mid = (l + r) / 2;
            if (nums[mid] == target) {
                return true;
            }
            if (nums[l] == nums[mid] && nums[mid] == nums[r]) {
                ++l;
                --r;
            } else if (nums[l] <= nums[mid]) {
                if (nums[l] <= target && target < nums[mid]) {
                    r = mid - 1;
                } else {
                    l = mid + 1;
                }
            } else {
                if (nums[mid] < target && target <= nums[n - 1]) {
                    l = mid + 1;
                } else {
                    r = mid - 1;
                }
            }
        }
        return false;
    }
};
```

## 删除排序链表中的重复元素

### 一次遍历

由于给定的链表是排好序的，因此**重复的元素在链表中出现的位置是连续的**，因此我们只需要对链表进行一次遍历，就可以删除重复的元素。

具体地，我们从指针 $\textit{cur}$ 指向链表的头节点，随后开始对链表进行遍历。如果当前 $\textit{cur}$ 与 $\textit{cur.next}$ 对应的元素相同，那么我们就将 $\textit{cur.next}$ 从链表中移除；否则说明链表中已经不存在其它与 $\textit{cur}$ 对应的元素相同的节点，因此可以将 $\textit{cur}$ 指向 $\textit{cur.next}$。

当遍历完整个链表之后，我们返回链表的头节点即可。

**细节：**当我们遍历到链表的最后一个节点时，$\textit{cur.next}$ 为空节点，如果不加以判断，访问 $\textit{cur.next}$ 对应的元素会产生运行错误。因此我们只需要遍历到链表的最后一个节点，而不需要遍历完整个链表。

``` c++
class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        if (!head) {
            return head;
        }

        ListNode* cur = head;
        while (cur->next) {
            if (cur->val == cur->next->val) {
                cur->next = cur->next->next;
            }
            else {
                cur = cur->next;
            }
        }

        return head;
    }
};
```

## 删除排序链表中的重复元素2

### 一次遍历

由于给定的链表是排好序的，因此**重复的元素在链表中出现的位置是连续的**，因此我们只需要对链表进行一次遍历，就可以删除重复的元素。由于链表的头节点可能会被删除，因此我们需要额外使用一个哑节点（dummy node）指向链表的头节点。

具体地，我们从指针 $\textit{cur}$ 指向链表的哑节点，随后开始对链表进行遍历。如果当前 $\textit{cur.next}$ 与 $\textit{cur.next.next}$ 对应的元素相同，那么我们就需要将 $\textit{cur.next}$ 以及所有后面拥有相同元素值的链表节点全部删除。我们记下这个元素值 $x$，随后不断将 $\textit{cur.next}$ 从链表中移除，直到 $\textit{cur.next}$ 为空节点或者其元素值不等于 $x$ 为止。此时，我们将链表中所有元素值为 $x$ 的节点全部删除。

如果当前 $\textit{cur.next}$ 与 $\textit{cur.next.next}$ 对应的元素不相同，那么说明链表中只有一个元素值为 $\textit{cur.next}$ 的节点，那么我们就可以将 $\textit{cur}cur 指向 \textit{cur.next}$。

当遍历完整个链表之后，我们返回链表的的哑节点的下一个节点 $\textit{dummy.next}$ 即可。

**细节：**需要注意 $\textit{cur.next}$ 以及 $\textit{cur.next.next}$ 可能为空节点，如果不加以判断，可能会产生运行错误。

``` c++
class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        if (!head) {
            return head;
        }
        
        ListNode* dummy = new ListNode(0, head);

        ListNode* cur = dummy;
        while (cur->next && cur->next->next) {
            if (cur->next->val == cur->next->next->val) {
                int x = cur->next->val;
                while (cur->next && cur->next->val == x) {
                    cur->next = cur->next->next;
                }
            }
            else {
                cur = cur->next;
            }
        }

        return dummy->next;
    }
};
```

**注意：**上面两题的 $\texttt{C++}$ 代码中并没有释放被删除的链表节点以及哑节点的空间。如果在面试中遇到本题，读者需要针对这一细节与面试官进行沟通。

## 柱状图中的最大矩形

### 题目

给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。求在该柱状图中，能够勾勒出来的矩形的最大面积。

### 前言

我们可以考虑枚举矩形的宽和高，其中「宽」表示矩形贴着柱状图底边的宽度，「高」表示矩形在柱状图上的高度。

- 如果我们枚举「宽」，我们可以使用两重循环枚举矩形的左右边界以固定宽度 $w$，此时矩形的高度 $h$，就是所有包含在内的柱子的「最小高度」，对应的面积为 $w * h$。$C++$ 代码如下：

``` c++
class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        int n = heights.size();
        int ans = 0;
        // 枚举左边界
        for (int left = 0; left < n; ++left) {
            int minHeight = INT_MAX;
            // 枚举右边界
            for (int right = left; right < n; ++right) {
                // 确定高度
                minHeight = min(minHeight, heights[right]);
                // 计算面积
                ans = max(ans, (right - left + 1) * minHeight);
            }
        }
        return ans;
    }
};
```

- 如果我们枚举「高」，我们可以使用一重循环枚举某一根柱子，将其固定为矩形的高度 $h$。随后我们从这跟柱子开始向两侧延伸，直到遇到高度小于 $h$ 的柱子，就确定了矩形的左右边界。如果左右边界之间的宽度为 $w$，那么对应的面积为 $w * h$。$C++$ 代码如下：

``` c++
class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        int n = heights.size();
        int ans = 0;
        for (int mid = 0; mid < n; ++mid) {
            // 枚举高
            int height = heights[mid];
            int left = mid, right = mid;
            // 确定左右边界
            while (left - 1 >= 0 && heights[left - 1] >= height) {
                --left;
            }
            while (right + 1 < n && heights[right + 1] >= height) {
                ++right;
            }
            // 计算面积
            ans = max(ans, (right - left + 1) * height);
        }
        return ans;
    }
};
```

可以发现，这两种暴力方法的时间复杂度均为 $O(N^2)$，会超出时间限制，我们必须要进行优化。考虑到枚举「宽」的方法使用了两重循环，本身就已经需要 $O(N^2)$ 的时间复杂度，不容易优化，因此我们可以考虑优化只使用了一重循环的枚举「高」的方法。。

### 单调栈

归纳一下枚举「高」的方法：

- 首先我们枚举某一根柱子 $i$ 作为高 $h = \textit{heights}[i]$；
- 随后我们需要进行向左右两边扩展，使得扩展到的柱子的高度均不小于 $h$。换句话说，我们需要找到**左右两侧最近的高度小于 $h$ 的柱子**，这样这两根柱子之间（不包括其本身）的所有柱子高度均不小于 $h$，并且就是 $i$ 能够扩展到的最远范围。


我们先来看看如何求出**一根柱子的左侧且最近的小于其高度的柱子**。除了根据「前言」部分暴力地进行枚举之外，我们可以通过如下的一个结论来深入地进行思考：

> 对于两根柱子 $j_0$ 以及 $j_1$，如果 $j_0 < j_1$ 并且 $height[j_0] \geq height[j_1]$，那么对于 任意的在它们之后出现的柱子 $i$ ($j_1 < i $)，$j_0$ 一定不会是 $i$ 左侧且最近的小于其高度的柱子。

换句话说，如果有两根柱子 $j_0$ 和 $j_1$，其中 $j_0$ 在 $j_1$ 的左侧，并且 $j_0$ 的高度大于等于 $j_1$， 那么在后面的柱子 $i$ 向左找小于其高度的柱子时， $j_1$ 会挡住 $j_0$，$j_0$ 就不会作为答案了。

这样以来，我们可以对数组从左向右进行遍历，同时维护一个「可能作为答案」的数据结构，其中按照从小到大的顺序存放了一些 $j$ 值。根据上面的结论，如果我们存放了 $j_0, j_1, \cdots, j_s$，那么一定有 $\textit{height}[j_0] < \textit{height}[j_1] < \cdots < \textit{height}[j_s]$，因为如果有两个相邻的 $j$ 值对应的高度不满足 $<$ 关系，那么后者会「挡住」前者，前者就不可能作为答案了。

当我们枚举到第 $i$ 根柱子时，我们的数据结构中存放了 $j_0, j_1, \cdots, j_s$ ，如果第 $i$ 根柱子左侧且最近的小于其高度的柱子为 $j_i$，那么必然有

$$
height[j_0]<height[j_1]<⋯<height[j_i]<height[i]≤height[j_{i+1}]<⋯<height[j_s]
$$
我们在枚举到第 $i$ 根柱子的时候，就可以先把所有高度大于等于 $\textit{height}[i]$ 的 $j$ 值全部移除，剩下的 $j$ 值中高度最高的即为答案。在这之后，我们将 $i$ 放入数据结构中，开始接下来的枚举。此时，我们需要使用的数据结构也就呼之欲出了，它就是**栈**。

- 栈中存放了 $j$ 值。从栈底到栈顶，$j$ 的值严格单调递增，同时对应的高度值也严格单调递增；

- 当我们枚举到第 $i$ 根柱子时，我们从栈顶不断地移除 $\textit{height}[j] \geq \textit{height}[i]$ 的 $j$ 值。在移除完毕后，栈顶的 $j$ 值就一定满足 $\textit{height}[j] < \textit{height}[i]$，此时 $j$ 就是 $i$ 左侧且最近的小于其高度的柱子。

  - 这里会有一种特殊情况。如果我们移除了栈中所有的 $j$ 值，那就说明 $i$ 左侧所有柱子的高度都大于 $\textit{height}[i]$，那么我们可以认为 $i$ 左侧且最近的小于其高度的柱子在位置 $j=-1$，它是一根「虚拟」的、高度无限低的柱子。这样的定义不会对我们的答案产生任何的影响，我们也称这根「虚拟」的柱子为**哨兵**。


- 将 $i$ 放入栈顶

栈中存放的元素具有单调性，这就是经典的数据结构**单调栈**了。

### 分析

> 每一个位置只会入栈一次（在枚举到它时），并且最多出栈一次。

因此当我们从左向右/总右向左遍历数组时，对栈的操作的次数就为 $O(N)$。所以单调栈的总时间复杂度为 $O(N)$。

``` c++
class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        int n = heights.size();
        vector<int> left(n), right(n);
        
        stack<int> mono_stack;
        for (int i = 0; i < n; ++i) {
            while (!mono_stack.empty() && heights[mono_stack.top()] >= heights[i]) {
                mono_stack.pop();
            }
            left[i] = (mono_stack.empty() ? -1 : mono_stack.top());
            mono_stack.push(i);
        }

        mono_stack = stack<int>();
        for (int i = n - 1; i >= 0; --i) {
            while (!mono_stack.empty() && heights[mono_stack.top()] >= heights[i]) {
                mono_stack.pop();
            }
            right[i] = (mono_stack.empty() ? n : mono_stack.top());
            mono_stack.push(i);
        }
        
        int ans = 0;
        for (int i = 0; i < n; ++i) {
            ans = max(ans, (right[i] - left[i] - 1) * heights[i]);
        }
        return ans;
    }
};
```

### 单调栈+常数优化

在方法一中，我们首先从左往右对数组进行遍历，借助单调栈求出了每根柱子的左边界，随后从右往左对数组进行遍历，借助单调栈求出了每根柱子的右边界。那么我们是否可以只遍历一次就求出答案呢？

答案是可以的。在方法一中，我们在对位置 $i$ 进行入栈操作时，确定了它的左边界。从直觉上来说，与之对应的我们在对位置 $i$ 进行出栈操作时可以确定它的右边界！仔细想一想，这确实是对的。当位置 $i$ 被弹出栈时，说明此时遍历到的位置 $i_0$ 的高度**小于等于** $\textit{height}[i]$，并且在 $i_0$ 与 $i$ 之间没有其他高度小于等于 $\textit{height}[i]$ 的柱子。这是因为，如果在 $i$ 和 $i_0$ 之间还有其它位置的高度小于等于 $\textit{height}[i]$ 的，那么在遍历到那个位置的时候，$i$ 应该已经被弹出栈了。所以位置 $i_0$ 就是位置 $i$ 的右边界。

**注意：**我们需要的是「一根柱子的左侧且最近的**小于**其高度的柱子」，但这里我们求的是**小于等于**，那么会造成什么影响呢？答案是：我们确实无法求出正确的右边界，但对最终的答案没有任何影响。这是因为在答案对应的矩形中，如果有若干个柱子的高度都等于矩形的高度，那么**最右侧的那根柱子是可以求出正确的右边界的**，而我们没有对求出左边界的算法进行任何改动，因此最终的答案还是可以从最右侧的与矩形高度相同的柱子求得的。

在遍历结束后，栈中仍然有一些位置，这些位置对应的右边界就是位置为 $n$ 的「哨兵」。我们可以将它们依次出栈并更新右边界，也可以在初始化右边界数组时就将所有的元素的值置为 $n$。

``` c++
class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        int n = heights.size();
        vector<int> left(n), right(n, n);
        
        stack<int> mono_stack;
        for (int i = 0; i < n; ++i) {
            while (!mono_stack.empty() && heights[mono_stack.top()] >= heights[i]) {
                right[mono_stack.top()] = i;
                mono_stack.pop();
            }
            left[i] = (mono_stack.empty() ? -1 : mono_stack.top());
            mono_stack.push(i);
        }
        
        int ans = 0;
        for (int i = 0; i < n; ++i) {
            ans = max(ans, (right[i] - left[i] - 1) * heights[i]);
        }
        return ans;
    }
};
```

## 最大矩形

### 题目

给定一个仅包含 `0` 和 `1` 、大小为 $rows \times cols$ 的二维二进制矩阵，找出只包含 `1` 的最大矩形，并返回其面积。

### 使用柱状图的优化暴力方法

最原始地，我们可以列举每个可能的矩形。我们枚举矩形所有可能的左上角坐标和右下角坐标，并检查该矩形是否符合要求。然而该方法的时间复杂度过高。

我们首先计算出矩阵的每个元素的左边连续 $1$ 的数量，使用二维数组 $\textit{left}$ 记录，其中 $\textit{left}[i][j]$ 为矩阵第 $i$ 行第 $j$ 列元素的左边连续 $1$ 的数量。

随后，对于矩阵中任意一个点，我们枚举以该点为右下角的全 $1$ 矩形。

具体而言，当考察以 $\textit{matrix}[i][j]$ 为右下角的矩形时，我们枚举满足 $0 \le k \le i$ 的所有可能的 $k$，此时矩阵的最大宽度就为
$$
left[i][j],left[i−1][j],…,left[k][j]
$$
的最小值。

我们预计算最大宽度的方法事实上将输入转化成了一系列的柱状图，我们针对每个柱状图计算最大面积。

<img src="https://assets.leetcode-cn.com/solution-static/85/3_1.png" width="750" height="400">

``` c++
class Solution {
public:
    int maximalRectangle(vector<vector<char>>& matrix) {
        int m = matrix.size();
        if (m == 0) {
            return 0;
        }
        int n = matrix[0].size();
        vector<vector<int>> left(m, vector<int>(n, 0));

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == '1') {
                    left[i][j] = (j == 0 ? 0: left[i][j - 1]) + 1;
                }
            }
        }

        int ret = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == '0') {
                    continue;
                }
                int width = left[i][j];
                int area = width;
                for (int k = i - 1; k >= 0; k--) {
                    width = min(width, left[k][j]);
                    area = max(area, (i - k + 1) * width);
                }
                ret = max(ret, area);
            }
        }
        return ret;
    }
};
```

- 时间复杂度：$O(m^2n)$，其中 $m$ 和 $n$ 分别是矩阵的行数和列数。$O(mn)+O(mn)*O(m)=O(m^2n)$。

### 单调栈

方法一中，将输入拆分成一系列的柱状图。为了计算矩形的最大面积，我们只需要计算每个柱状图中的最大面积，并找到全局最大值。可以使用上一题**柱状图中的最大矩形**的单调栈算法。

``` c++
class Solution {
public:
    int maximalRectangle(vector<vector<char>>& matrix) {
        int m = matrix.size();
        if (m == 0) {
            return 0;
        }
        int n = matrix[0].size();
        vector<vector<int>> left(m, vector<int>(n, 0));

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == '1') {
                    left[i][j] = (j == 0 ? 0: left[i][j - 1]) + 1;
                }
            }
        }

        int ret = 0;
        for (int j = 0; j < n; j++) { // 对于每一列，使用基于柱状图的方法
            vector<int> up(m, 0), down(m, 0);

            stack<int> stk;
            for (int i = 0; i < m; i++) {
                while (!stk.empty() && left[stk.top()][j] >= left[i][j]) {
                    stk.pop();
                }
                up[i] = stk.empty() ? -1 : stk.top();
                stk.push(i);
            }
            stk = stack<int>();
            for (int i = m - 1; i >= 0; i--) {
                while (!stk.empty() && left[stk.top()][j] >= left[i][j]) {
                    stk.pop();
                }
                down[i] = stk.empty() ? m : stk.top();
                stk.push(i);
            }

            for (int i = 0; i < m; i++) {
                int height = down[i] - up[i] - 1;
                int area = height * left[i][j];
                ret = max(ret, area);
            }
        }
        return ret;
    }
};
```

- 时间复杂度：$O(mn)$

## 分割链表

### 模拟

我们只需维护两个链表 $\textit{small}$ 和 $\textit{large}$ 即可，$\textit{small}$ 链表按顺序存储所有小于 $x$ 的节点，$\textit{large}$ 链表按顺序存储所有大于等于 $x$ 的节点。遍历完原链表后，我们只要将 $\textit{small}$ 链表尾节点指向 $\textit{large}$ 链表的头节点即能完成对链表的分隔。

为了实现上述思路，我们设 $\textit{smallHead}$ 和 $\textit{largeHead}$ 分别为两个链表的哑节点，即它们的 $\textit{next}$ 指针指向链表的头节点，这样做的目的是为了更方便地处理**头节点为空的边界条件**。同时设 $\textit{small}$ 和 $\textit{large}$ 节点指向当前链表的末尾节点。开始时 $\textit{smallHead}=\textit{small},\textit{largeHead}=\textit{large}$。随后，从前往后遍历链表，判断当前链表的节点值是否小于 $x$，如果小于就将 $\textit{small}$ 的 $\textit{next}$ 指针指向该节点，否则将 $\textit{large}$ 的 $\textit{next}$ 指针指向该节点。

``` c++
class Solution {
public:
    ListNode* partition(ListNode* head, int x) {
        ListNode* small = new ListNode(0);
        ListNode* smallHead = small;
        ListNode* large = new ListNode(0);
        ListNode* largeHead = large;
        while (head != nullptr) {
            if (head->val < x) {
                small->next = head;
                small = small->next;
            } else {
                large->next = head;
                large = large->next;
            }
            head = head->next;
        }
        large->next = nullptr;
        small->next = largeHead->next;
        return smallHead->next;
    }
};
```

## 扰乱字符串

### 动态规划

显然「扰乱字符串」的关系是具有**对称性**的，即如果 $s_1$ 是 $s_2$ 的扰乱字符串，那么 $s_2$ 也是 $s_1$  的扰乱字符串。为了叙述方便，我们称这种情况下，$s_1$ 和 $s_2$ 是**和谐**的。

如何判断 $s_1$ 和 $s_2$ 是否**和谐**：

- 如果 $s_1 = s_2$，那么它们是和谐的;
- 如果 $s_1$ 和 $s_2$ 长度不一致，则一定不和谐;
- 如果 $s_1$ 中某个字符 $c$ 出现了 $x_1$ 次，而 $c$ 在 $s_2$  中出现了 $x_2$ 次，且 $x_1 \neq x_2$ ，那么它们一定不是和谐的。

对于剩下的情况，我们该如何判断呢？我们可以从 $s_1$ 的分割方法入手。假设 $s_1$  作为根节点时被分割成了 $l(s_1)$ 以及 $r(s_1)$ 两个子串，那么：

- 如果 $l(s_1)$ 和 $r(s_1)$ 没有被交换，那么 $s_2$ 需要存在一种分割方法 $s_2 = l(s_2)+r(s_2)$，使得 $l(s_1)$ 和 $l(s_2)$ 是和谐的，并且 $r(s_1)$ 和 $r(s_2)$ 也是和谐的
- 如果 $l(s_1)$ 和 $r(s_1)$ 被交换了，那么 $s_2$ 需要存在一种分割方法 $s_2 = l(s_2)+r(s_2)$，使得 $l(s_1)$ 和 $r(s_2)$ 是和谐的，并且 $r(s_1)$ 和 $l(s_2)$ 也是和谐的

<img src="https://assets.leetcode-cn.com/solution-static/87/1.png" width="800" height="400" >

 这样一来，我们就把原本需要解决的问题划分成了两个本质相同，但规模更小的子问题，因此可以考虑使用动态规划解决。

设 $f(s_1, s_2)$ 表示 $s_1$ 和 $s_2$ 是否和谐，我们可以写出状态转移方程
$$
f(s_1, s_2) =\begin{cases}True, & s_1=s_2 \\ False, & 某个字符c，它在s_1和s_2中的出现次数不同 \end{cases}
$$
设 $s_1$，$s_2$ 的长度为 $n$，我们用 $s_1(x,y)$ 表示从 $s_1$ 的第 $x$ 个字符（从 $0$ 开始编号）开始，长度为 $y$ 的子串。由于分割出的两个字符串不能为空串，那么其中一个字符串就是 $s_1(0, i)$，另一个字符串是 $s_1(i, n-i)$。

- 对于 $l(s_1)$ 和 $r(s_1)$ 没有被交换的情况，$s_2$ 同样需要被分割为 $s_2(0, i)$ 以及 $s_2(i, n-i)$。否则长度不同的字符串是不可能「和谐」的。因此我们可以写出状态转移方程：

$$
f(s_1, s_2) = \bigvee_{i=1}^{n-1}(f(s_1(0, i),s_2(0,i)) \wedge f(s1(i, n-i),s_2(i,n-i)))
$$

- 对于 $l(s_1)$ 和 $r(s_1)$ 被交换的情况，$s_2$ 需要被分割为 $s_2(0, n-i)$ 以及 $s_2(n-i, i)$，这样对应的长度才会相同。因此我们可以写出状态转移方程：

$$
f(s_1, s_2) = \bigvee_{i=1}^{n-1}(f(s_1(0, i),s_2(n-i,i)) \wedge f(s1(i, n-i),s_2(0,n-i)))
$$

将上面两种状态转移方程用 $\vee$ 或运算拼在一起，即可得到最终的状态转移方程。

### 细节

1. 在进行状态转移时，我们需要先计算出较短的字符串对应的 $f$ 值，再去转移计算出较长的字符串对应的 $f$ 值，这是因为我们需要保证在计算 $f(s_1, s_2)$ 时，所有它们的子串对应的状态都需要被计算过。我们可以考虑使用**记忆化搜索自顶向下**地进行动态规划，这样我们只需要用题目中给定的两个原始字符串开始，递归地计算所有的 $f$ 值，而无需考虑计算顺序。

2. 由于我们使用记忆化搜索，因此我们需要把 $s_1$ 和 $s_2$ 作为参数传入记忆化搜索使用的递归函数。这样一来，在递归传递参数的过程中，会使用到大量字符串的切片、拷贝等操作，使得时空复杂度不那么优。本题中，由于给定原始字符串的长度不超过 $30$，因此不会产生太大的影响，但我们还是要尽可能对代码进行优化。

   一种通用的优化方法是，我们将状态变更为 $f(i_1, i_2, \textit{length})$，表示第一个字符串是原始字符串从第 $i_1$ 个字符开始，长度为 \textit{length}length 的子串，第二个字符串是原始字符串从第 $i_2$ 个字符开始，长度为 $\textit{length}$ 的子串。可以发现，我们只是改变了表达 $s_1$ 和 $s_2$ 的方式，但此时我们只需要在递归时传递三个整数类型的变量，省去了字符串的操作；

``` c++
class Solution {
private:
    // 记忆化搜索存储状态的数组
    // -1 表示 false，1 表示 true，0 表示未计算
    int memo[30][30][31];
    string s1, s2;

public:
    bool checkIfSimilar(int i1, int i2, int length) {
        unordered_map<int, int> freq;
        for (int i = i1; i < i1 + length; ++i) {
            ++freq[s1[i]];
        }
        for (int i = i2; i < i2 + length; ++i) {
            --freq[s2[i]];
        }
        if (any_of(freq.begin(), freq.end(), [](const auto& entry) {return entry.second != 0;})) {
            return false;
        }
        return true;
    }

    // 第一个字符串从 i1 开始，第二个字符串从 i2 开始，子串的长度为 length，是否和谐
    bool dfs(int i1, int i2, int length) {
        if (memo[i1][i2][length]) {
            return memo[i1][i2][length] == 1;
        }

        // 判断两个子串是否相等
        if (s1.substr(i1, length) == s2.substr(i2, length)) {
            memo[i1][i2][length] = 1;
            return true;
        }

        // 判断是否存在字符 c 在两个子串中出现的次数不同
        if (!checkIfSimilar(i1, i2, length)) {
            memo[i1][i2][length] = -1;
            return false;
        }
        
        // 枚举分割位置
        for (int i = 1; i < length; ++i) {
            // 不交换的情况
            if (dfs(i1, i2, i) && dfs(i1 + i, i2 + i, length - i)) {
                memo[i1][i2][length] = 1;
                return true;
            }
            // 交换的情况
            if (dfs(i1, i2 + length - i, i) && dfs(i1 + i, i2, length - i)) {
                memo[i1][i2][length] = 1;
                return true;
            }
        }

        memo[i1][i2][length] = -1;
        return false;
    }

    bool isScramble(string s1, string s2) {
        memset(memo, 0, sizeof(memo));
        this->s1 = s1;
        this->s2 = s2;
        return dfs(0, 0, s1.size());
    }
};
```

## 合并两个有序数组

## 逆向双指针

$ nums1 $ 的后半部分是空的，可以直接覆盖而不会影响结果。因此可以指针设置为从后向前遍历，每次取两者之中的较大者放进 $nums1$ 的最后面。严格来说，在此遍历过程中的任意一个时刻，$nums1$ 数组中有 $m-p_1-1$ 个元素被放入 $nums1$ 的后半部，$nums2$ 数组中有 $m-p_2-1$ 个元素被放入 $nums1$ 的后半部，而在指针 $p_1$ 的后面，$nums1$ 数组有 $m+n-p_1-1$ 个位置。由于
$$
m+n-p_1-1 \geq m-p_1-1+n-p_2-1
$$
等价于
$$
p_2 \geq -1
$$
永远成立。因此 $p_1$ 后面的位置永远足够容纳被插入的元素，不会产生 $p_1$ 的元素被覆盖的情况。

``` c+
class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        int p1 = m - 1, p2 = n - 1;
        int tail = m + n - 1;
        int cur;
        while (p1 >= 0 || p2 >= 0) {
            if (p1 == -1) {
                cur = nums2[p2--];
            } else if (p2 == -1) {
                cur = nums1[p1--];
            } else if (nums1[p1] > nums2[p2]) {
                cur = nums1[p1--];
            } else {
                cur = nums2[p2--];
            }
            nums1[tail--] = cur;
        }
    }
};
```

## 格雷编码

### 题目

格雷编码是一个二进制数字系统，在该系统中，两个连续的数值仅有一个位数的差异。给定一个代表编码总位数的非负整数 n，打印其格雷编码序列。即使有多个不同答案，你也只需要返回其中一种。格雷编码序列必须以 0 开头。

### 镜像反射法

<img src="https://pic.leetcode-cn.com/d0df7e038c396acf7c5283e8080963ecefe2ab37d4b607982eb3e40b1e5ee03b-Picture3.png" width="600" height="400">

``` c++
class Solution {
public:

    void encode(vector<int> &ans, int n) {
        if (n == 0)
            ans.push_back(0);
        else
        {
            encode(ans, n-1);
            int len = ans.size();
            for (int i = len-1; i >= 0; --i) {
                ans.push_back(ans[i]+pow(2,n-1));
            }
        }
        
    }

    vector<int> grayCode(int n) {
        vector<int> ans;
        encode(ans, n);

        return ans;
    }
};
```

### 公式

二进制转成格雷码有一个公式。

<img src="https://pic.leetcode-cn.com/1013850d7f6c8cf1d99dc0ac3292264b74f6a52d84e0215f540c80952e184f41-image.png" width="600" height="400">

所以我们遍历 $0$ 到 $2^n-1$，然后利用公式转换即可。即最高位保留，其它位是当前位和它的高一位进行异或操作。

``` c++
class Solution {
public:
    vector<int> grayCode(int n) {
        vector<int> ans;
        for(int binary=0; binary < 1<<n; ++binary) {
            ans.push_back(binary^binary >> 1);
        }

        return ans;
    }
};
```

## 子集2

### 递归

递归时，若发现没有选择上一个数，且当前数字与上一个数相同，则可以跳过当前生成的子集。

``` c++
class Solution {
public:

    void dfs(vector<vector<int>> &ans, vector<int> &tmp, vector<int> &nums, int p) {
        if (p == nums.size()) {
            ans.push_back(tmp);
            return;
        }

        tmp.push_back(nums[p]);
        dfs(ans, tmp, nums, p+1);
        tmp.pop_back();
        while(p+1 < nums.size() && nums[p+1] == nums[p])
            ++p;
        dfs(ans, tmp, nums, p+1);
    }

    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        sort(nums.begin(), nums.end());

        vector<int> tmp;
        vector<vector<int>> ans;

        dfs(ans, tmp, nums, 0);

        return ans;
    }
};
```

### 迭代

考虑数组 $[1,2,2]$，选择前两个数，或者第一、三个数，都会得到相同的子集。

也就是说，对于当前选择的数 $x$，若前面有与其相同的数 $y$，且没有选择 $y$，此时包含 $x$ 的子集，必然会出现在包含 $y$ 的所有子集中。

我们可以通过判断这种情况，来避免生成重复的子集。代码实现时，可以先将数组排序；迭代时，若发现没有选择上一个数，且当前数字与上一个数相同，则可以跳过当前生成的子集。

``` c++
class Solution {
public:
    vector<int> t;
    vector<vector<int>> ans;

    vector<vector<int>> subsetsWithDup(vector<int> &nums) {
        sort(nums.begin(), nums.end());
        int n = nums.size();
        for (int mask = 0; mask < (1 << n); ++mask) {
            t.clear();
            bool flag = true;
            for (int i = 0; i < n; ++i) {
                if (mask & (1 << i)) {
                    if (i > 0 && (mask >> (i - 1) & 1) == 0 && nums[i] == nums[i - 1]) {
                        flag = false;
                        break;
                    }
                    t.push_back(nums[i]);
                }
            }
            if (flag) {
                ans.push_back(t);
            }
        }
        return ans;
    }
};
```

- 时间复杂度：$O(n\times2^n)$，其中 $n$ 是数组 $\textit{nums}$ 的长度。排序的时间复杂度为 $O(n \log n)$。最坏情况下 $\textit{nums}$中无重复元素，需要枚举其所有 $2^n$ 个子集，每个子集加入答案时需要拷贝一份，耗时 $O(n)$，一共需要 $O(n \times 2^n)+O(n)=O(n \times 2^n)$ 的时间来构造子集。

## 解码方法

### 动态规划

对于给定的字符串 $s$，设它的长度为 $n$，其中的字符从左到右依次为 $s[1], s[2], \cdots, s[n]$。我们可以使用动态规划的方法计算出字符串 $s$ 的解码方法数。

具体地，设 $f_i$ 表示字符串 $s$ 的前 $i$ 个字符 $s[1..i]$ 的解码方法数。在进行状态转移时，我们可以考虑最后一次解码使用了 $s$ 中的哪些字符，那么会有下面的两种情况：

- 第一种情况是我们使用了一个字符，即 $s[i]$ 进行解码，那么只要 $s[i] \neq 0$，它就可以被解码成 $\text{A} \sim \text{I}$ 中的某个字母。由于剩余的前 $i-1$ 个字符的解码方法数为 $f_{i-1}$ ，因此我们可以写出状态转移方程：

$$
f_i=f_{i-1},其中s[i] \neq 0
$$

- 第二种情况是我们使用了两个字符，即 $s[i-1]$ 和 $s[i]$ 进行编码。与第一种情况类似，$s[i-1]$ 不能等于 $0$，并且 $s[i-1]$ 和 $s[i]$ 组成的整数必须小于等于 $26$，这样它们就可以被解码成 $\text{J} \sim \text{Z}$ 中的某个字母。由于剩余的前 $i-2$ 个字符的解码方法数为 $f_{i-2}$，因此我们可以写出状态转移方程：

$$
f_i=f_{i-2},其中s[i-1] \neq 0并且10 \cdot s[i-1] + s[i] \leq26
$$

​		需要注意的是，只有$s > 1$ 时才能进行转移，否则 $s[i-1]$ 不存在

将上面的两种状态转移方程在对应的条件满足时进行累加，即可得到 $f_i$ 的值。在动态规划完成后，最终的答案即为 $f_n$ 。

**细节：**动态规划的边界条件是：
$$
f_0=1
$$
 即**空字符串可以有 1 种解码方法，解码出一个空字符串。**

``` c++
class Solution {
public:
    int numDecodings(string s) {
        int n = s.size();
        vector<int> f(n + 1);
        f[0] = 1;
        for (int i = 1; i <= n; ++i) {
            if (s[i - 1] != '0') {
                f[i] += f[i - 1];
            }
            if (i > 1 && s[i - 2] != '0' && ((s[i - 2] - '0') * 10 + (s[i - 1] - '0') <= 26)) {
                f[i] += f[i - 2];
            }
        }
        return f[n];
    }
};
```

**滚动数组优化：**注意到在状态转移方程中，$f_i$ 的值仅与 $f_{i-1}$ 和 $f_{i-2}$ 有关，因此我们可以使用三个变量进行状态转移，省去数组的空间。

```c++
class Solution {
public:
    int numDecodings(string s) {
        int n = s.size();
        // a = f[i-2], b = f[i-1], c = f[i]
        int a = 0, b = 1, c;
        for (int i = 1; i <= n; ++i) {
            c = 0;
            if (s[i - 1] != '0') {
                c += b;
            }
            if (i > 1 && s[i - 2] != '0' && ((s[i - 2] - '0') * 10 + (s[i - 1] - '0') <= 26)) {
                c += a;
            }
            tie(a, b) = {b, c};
        }
        return c;
    }
};
```

## 反转链表2

### 穿针引线

<img src="https://pic.leetcode-cn.com/1615105129-iUPoGi-image.png" width="800" height="200">

反转 `left` 到 `right` 部分以后，再拼接起来。我们还需要记录 `left` 的前一个节点，和 `right` 的后一个节点。如图所示：

<img src="https://pic.leetcode-cn.com/1615105150-pfWiGq-image.png" width="500" height="250">

**算法步骤：**

- 先将待反转的区域反转；
- 把 `pre` 的 `next` 指针指向反转以后的链表头节点，把反转以后的链表的尾节点的 `next` 指针指向 `succ`。

<img src="https://pic.leetcode-cn.com/1615105168-ZQRZew-image.png" width=600>

``` c++
class Solution {
private:
    void reverseLinkedList(ListNode *head) {
        // 也可以使用递归反转一个链表
        ListNode *pre = nullptr;
        ListNode *cur = head;

        while (cur != nullptr) {
            ListNode *next = cur->next;
            cur->next = pre;
            pre = cur;
            cur = next;
        }
    }

public:
    ListNode *reverseBetween(ListNode *head, int left, int right) {
        // 因为头节点有可能发生变化，使用虚拟头节点可以避免复杂的分类讨论
        ListNode *dummyNode = new ListNode(-1);
        dummyNode->next = head;

        ListNode *pre = dummyNode;
        // 第 1 步：从虚拟头节点走 left - 1 步，来到 left 节点的前一个节点
        // 建议写在 for 循环里，语义清晰
        for (int i = 0; i < left - 1; i++) {
            pre = pre->next;
        }

        // 第 2 步：从 pre 再走 right - left + 1 步，来到 right 节点
        ListNode *rightNode = pre;
        for (int i = 0; i < right - left + 1; i++) {
            rightNode = rightNode->next;
        }

        // 第 3 步：切断出一个子链表（截取链表）
        ListNode *leftNode = pre->next;
        ListNode *curr = rightNode->next;

        // 注意：切断链接
        pre->next = nullptr;
        rightNode->next = nullptr;

        // 第 4 步：同第 206 题，反转链表的子区间
        reverseLinkedList(leftNode);

        // 第 5 步：接回到原来的链表中
        pre->next = rightNode;
        leftNode->next = curr;
        return dummyNode->next;
    }
};
```

### 头插法

方法一的缺点是：如果 `left` 和 `right` 的区域很大，恰好是链表的头节点和尾节点时，找到 `left` 和 `right` 需要遍历一次，反转它们之间的链表还需要遍历一次，虽然总的时间复杂度为 $O(N)$，但遍历了链表 $2$ 次。

一次遍历的整体思想是：在需要反转的区间里，每遍历到一个节点，让这个新节点来到反转部分的起始位置。下面的图展示了整个流程。

<img src="https://pic.leetcode-cn.com/1615105242-ZHlvOn-image.png" width="500" height="500">

下面我们具体解释如何实现。使用三个指针变量 `pre`、`curr`、`next` 来记录反转的过程中需要的变量，它们的意义如下：

- `curr`：指向待反转区域的第一个节点 `left`；
- `next`：永远指向 `curr` 的下一个节点，循环过程中，`curr` 变化以后 `next` 会变化；
- `pre`：永远指向待反转区域的第一个节点 `left` 的前一个节点，在循环过程中不变。

**操作步骤：**

- 先将 `curr` 的下一个节点记录为 `next`；
- 执行操作 ①：把 `curr` 的下一个节点指向 `next` 的下一个节点；
- 执行操作 ②：把 `next` 的下一个节点指向 `pre` 的下一个节点；
- 执行操作 ③：把` pre` 的下一个节点指向 `next`。

<img src="https://pic.leetcode-cn.com/1615105296-bmiPxl-image.png" width="500" height="200">

<img src="https://pic.leetcode-cn.com/1615105340-UBnTBZ-image.png" width="500" height="180">

<img src="https://pic.leetcode-cn.com/1615105353-PsCmzb-image.png" width="500" height="220">

<img src="https://pic.leetcode-cn.com/1615105364-aDIFqy-image.png" width="500" height="160">

<img src="https://pic.leetcode-cn.com/1615105376-jIyGwv-image.png" width="500" height="220">

<img src="https://pic.leetcode-cn.com/1615105395-EJQnMe-image.png" width="500" height="100">

``` c++
class Solution {
public:
    ListNode *reverseBetween(ListNode *head, int left, int right) {
        // 设置 dummyNode 是这一类问题的一般做法
        ListNode *dummyNode = new ListNode(-1);
        dummyNode->next = head;
        ListNode *pre = dummyNode;
        for (int i = 0; i < left - 1; i++) {
            pre = pre->next;
        }
        ListNode *cur = pre->next;
        ListNode *next;
        for (int i = 0; i < right - left; i++) {
            next = cur->next;
            cur->next = next->next;
            next->next = pre->next;
            pre->next = next;
        }
        return dummyNode->next;
    }
};
```

## 复原IP地址

### 回溯

设题目中给出的字符串为 $s$。我们用递归函数 $\textit{dfs}(\textit{segId}, \textit{segStart})$ 表示我们正在从 $s[\textit{segStart}]$ 的位置开始，搜索 IP 地址中的第 $\textit{segId}$ 段，其中 $\textit{segId} \in \{0, 1, 2, 3\}$。由于 IP 地址的每一段必须是 $[0, 255]$ 中的整数，因此我们从 $\textit{segStart}$ 开始，从小到大依次枚举当前这一段 IP 地址的结束位置 $\textit{segEnd}$。如果满足要求，就递归地进行下一段搜索，调用递归函数 $\textit{dfs}(\textit{segId} + 1, \textit{segEnd} + 1)$。

特别地，由于 IP 地址的每一段不能有前导零，因此如果 $s[\textit{segStart}]$ 等于字符 $0$，那么 IP 地址的第 $\textit{segId}$ 段只能为 $0$，需要作为特殊情况进行考虑。

在搜索的过程中，如果我们已经得到了全部的 $4$ 段 IP 地址（即 $\textit{segId} = 4$），并且遍历完了整个字符串（即 $\textit{segStart} = |s|$，其中 $|s|$ 表示字符串 $s$ 的长度），那么就复原出了一种满足题目要求的 IP 地址，我们将其加入答案。在其它的时刻，如果**提前**遍历完了整个字符串，那么我们需要结束搜索，回溯到上一步。

``` c++
class Solution {
private:
    static constexpr int SEG_COUNT = 4;

private:
    vector<string> ans;
    vector<int> segments;

public:
    void dfs(const string& s, int segId, int segStart) {
        // 如果找到了 4 段 IP 地址并且遍历完了字符串，那么就是一种答案
        if (segId == SEG_COUNT) {
            if (segStart == s.size()) {
                string ipAddr;
                for (int i = 0; i < SEG_COUNT; ++i) {
                    ipAddr += to_string(segments[i]);
                    if (i != SEG_COUNT - 1) {
                        ipAddr += ".";
                    }
                }
                ans.push_back(move(ipAddr));
            }
            return;
        }

        // 如果还没有找到 4 段 IP 地址就已经遍历完了字符串，那么提前回溯
        if (segStart == s.size()) {
            return;
        }

        // 由于不能有前导零，如果当前数字为 0，那么这一段 IP 地址只能为 0
        if (s[segStart] == '0') {
            segments[segId] = 0;
            dfs(s, segId + 1, segStart + 1);
        }

        // 一般情况，枚举每一种可能性并递归
        int addr = 0;
        for (int segEnd = segStart; segEnd < s.size(); ++segEnd) {
            addr = addr * 10 + (s[segEnd] - '0');
            if (addr > 0 && addr <= 0xFF) {
                segments[segId] = addr;
                dfs(s, segId + 1, segEnd + 1);
            } else {
                break;
            }
        }
    }

    vector<string> restoreIpAddresses(string s) {
        segments.resize(SEG_COUNT);
        dfs(s, 0, 0);
        return ans;
    }
};
```

- 时间复杂度：$O(3^{SEG\_COUNT}\times|s|)$。

## 不同二叉搜索树

### 题目

给你一个整数 $n$ ，求恰由 $n$ 个节点组成且节点值从 $1$ 到 $n$ 互不相同的 二叉搜索树 有多少种？返回满足题意的二叉搜索树的种数。

### 动态规划

**思路：**给定一个有序序列 $1 \cdots n$，为了构建出一棵二叉搜索树，我们可以遍历每个数字 $i$，将该数字作为树根，将 $1 \cdots (i-1)$ 序列作为左子树，将 $(i+1) \cdots n$ 序列作为右子树。接着我们可以按照同样的方式递归构建左子树和右子树。

在上述构建的过程中，由于根的值不同，因此我们能保证每棵二叉搜索树是唯一的。

由此可见，原问题可以分解成规模较小的两个子问题，且子问题的解可以复用。因此，我们可以想到使用**动态规划**来求解本题。

**算法：**

我们可以定义两个函数：

- $G(n)$: 长度为 $n$ 的序列能构成的不同二叉搜索树的个数。
- $F(i,n)$: 以 $i$ 为根、序列长度为 $n$ 的不同二叉搜索树个数 ($1 \leq i \leq n$)。

不同的二叉搜索树的总数 $G(n)$，是对遍历所有 $i$ ($1 \le i \le n$) 的 $F(i, n)$ 之和。换言之：
$$
G(n)=\sum_{i=1}^nF(i,n)
$$
对于边界情况，当序列长度为 $1$（只有根）或为 $0$（空树）时，只有一种情况，即：
$$
G(0)=1,G(1)=1
$$
给定序列 $1 \cdots n$，我们选择数字 $i$ 作为根，则根为 $i$ 的所有二叉搜索树的集合是左子树集合和右子树集合的**笛卡尔积**，对于笛卡尔积中的每个元素，加上根节点之后形成完整的二叉搜索树，如下图所示：

<img src="https://assets.leetcode-cn.com/solution-static/96/96_fig1.png" width="600" height="230">
$$
F(i,n)=G(i-1)G(n-i)
$$
将两个公式 结合，可以得到 $G(n)$ 的递归表达式：
$$
G(n)=\sum_{i=1}^nG(i-1)G(n-i)
$$
至此，我们从小到大计算 $G$ 函数即可，因为 $G(n)$ 的值依赖于 $G(0) \cdots G(n-1)$。

``` c++
class Solution {
public:
    int numTrees(int n) {
        vector<int> G(n + 1, 0);
        G[0] = 1;
        G[1] = 1;

        for (int i = 2; i <= n; ++i) {
            for (int j = 1; j <= i; ++j) {
                G[i] += G[j - 1] * G[i - j];
            }
        }
        return G[n];
    }
};
```

### 数学

事实上我们在方法一中推导出的 $G(n)$函数的值在数学上被称为**卡塔兰数** $C_n$。卡塔兰数更便于计算的定义如下:
$$
C_0=1,C_{n+1}=\frac{2(2n+1)}{n+2}C_n
$$

``` c++
class Solution {
public:
    int numTrees(int n) {
        long long C = 1;
        for (int i = 0; i < n; ++i) {
            C = C * 2 * (2 * i + 1) / (i + 2);
        }
        return (int)C;
    }
};
```

## 不同二叉搜索树2

### 题目

给定一个整数 $n$，生成所有由 $1 ... n$ 为节点所组成的 二叉搜索树 。

### 回溯

二叉搜索树关键的性质是根节点的值大于左子树所有节点的值，小于右子树所有节点的值，且左子树和右子树也同样为二叉搜索树。因此在生成所有可行的二叉搜索树的时候，假设当前序列长度为 $n$，如果我们枚举根节点的值为 $i$，那么根据二叉搜索树的性质我们可以知道左子树的节点值的集合为 [$1 \ldots i-1]$，右子树的节点值的集合为 $[i+1 \ldots n]$。而左子树和右子树的生成相较于原问题是一个序列长度缩小的子问题，因此我们可以想到用**回溯**的方法来解决这道题目。

我们定义 $generateTrees(start, end)$ 函数表示当前值的集合为 $[\textit{start},\textit{end}]$，返回序列 $[\textit{start},\textit{end}]$ 生成的所有可行的二叉搜索树。按照上文的思路，我们考虑枚举 $[\textit{start},\textit{end}]$ 中的值 $i$ 为当前二叉搜索树的根，那么序列划分为了 $[\textit{start},i-1]$ 和 $[i+1,\textit{end}]$ 两部分。我们递归调用这两部分，即 $generateTrees(start, i - 1)$ 和 $generateTrees(i + 1, end)$，获得所有可行的左子树和可行的右子树，那么最后一步我们只要从可行左子树集合中选一棵，再从可行右子树集合中选一棵拼接到根节点上，并将生成的二叉搜索树放入答案数组即可。

递归的入口即为 $generateTrees(1, n)$，出口为当 $\textit{start}>\textit{end}$ 的时候，当前二叉搜索树为空，返回空节点即可。

``` c++
class Solution {
public:
    vector<TreeNode*> generateTrees(int start, int end) {
        if (start > end) {
            return { nullptr };
        }
        vector<TreeNode*> allTrees;
        // 枚举可行根节点
        for (int i = start; i <= end; i++) {
            // 获得所有可行的左子树集合
            vector<TreeNode*> leftTrees = generateTrees(start, i - 1);

            // 获得所有可行的右子树集合
            vector<TreeNode*> rightTrees = generateTrees(i + 1, end);

            // 从左子树集合中选出一棵左子树，从右子树集合中选出一棵右子树，拼接到根节点上
            for (auto& left : leftTrees) {
                for (auto& right : rightTrees) {
                    TreeNode* currTree = new TreeNode(i);
                    currTree->left = left;
                    currTree->right = right;
                    allTrees.emplace_back(currTree);
                }
            }
        }
        return allTrees;
    }

    vector<TreeNode*> generateTrees(int n) {
        if (!n) {
            return {};
        }
        return generateTrees(1, n);
    }
};
```

## 交错字符串

### 题目

给定三个字符串 $s1$、$s2$、$s3$，请你帮忙验证 $s3$ 是否是由 $s1$ 和 $s2$ **交错**组成的。

两个字符串 s 和 t **交错**的定义与过程如下，其中每个字符串都会被分割成若干**非空**子字符串：

- $s = s_1 + s_2 + ... + s_n$
- $t = t_1 + t_2 + ... + t_m$
- $\|n - m\| \leq 1$
- 交错 是 $s_1 + t_1 + s_2 + t_2 + s_3 + t_3 + ...$ 或者 $t_1 + s_1 + t_2 + s_2 + t_3 + s_3 + ...$

### 动态规划

**双指针法错在哪里：**指针 $p_1$ 一开始指向 $s_1$ 的头部，指针 $p_2$ 一开始指向 $s_2$ 的头部，指针 $p_3$ 指向 $s_3$ 的头部，每次观察 $p_1$ 和 $p_2$ 指向的元素哪一个和 $p_3$ 指向的元素相等，相等则匹配并后移指针。样例就是一个很好的反例，用这种方法判断 $s_1 = {\rm aabcc}$，$s_2 = {\rm dbbca}$，$s_3 = {\rm aadbbcbcac}$ 时，得到的结果是 $\rm False$，实际应该是 $\rm True$。

**解决这个问题的正确方法是动态规划。**首先如果 $|s_1|+|s_2|\neq|s_3|$，那么 $s_3$ 必然不可能由 $s_1$ 和 $s_2$ 交错组成。在 $|s_1|+|s_2|\neq|s_3|$ 时，我们可以用**动态规划**来求解。定义 $f(i, j)$ 表示 $s_1$ 的前 $i$ 个元素和 $s_2$ 的前 $j$ 个元素是否能交错组成 $s_3$ 的前 $i + j$ 个元素。如果 $s_1$ 的第 $i$ 个元素和 $s_3$ 的第 $i + j$ 个元素相等，那么 $s_1$ 的前 $i$ 个元素和 $s_2$ 的前 $j$ 个元素是否能交错组成 $s_3$ 的前 $i + j$ 个元素取决于 $s_1$ 的前 $i - 1$ 个元素和 $s_2$ 的前 $j$ 个元素是否能交错组成 $s_3$ 的前 $i + j - 1$ 个元素，即此时 $f(i, j)$ 取决于 $f(i - 1, j)$，在此情况下如果 $f(i - 1, j)$ 为真，则 $f(i, j)$ 也为真。同样的，如果 $s_2$ 的第 $j$ 个元素和 $s_3$ 的第 $i + j$ 个元素相等并且 $f(i, j - 1)$ 为真，则 $f(i, j)$ 也为真。于是我们可以推导出这样的动态规划转移方程：
$$
f(i,j)=[f(i-1,j)~and~s_1(i-1)==s_3(p)]~or~[f(i,j-1)~and~s_2(j-1)==s3(p)]
$$
其中 $p=i+j-1$。边界条件为 $f(0,0)=true$。

``` c++
class Solution {
public:
    bool isInterleave(string s1, string s2, string s3) {
        auto f = vector < vector <int> > (s1.size() + 1, vector <int> (s2.size() + 1, false));

        int n = s1.size(), m = s2.size(), t = s3.size();

        if (n + m != t) {
            return false;
        }

        f[0][0] = true;
        for (int i = 0; i <= n; ++i) {
            for (int j = 0; j <= m; ++j) {
                int p = i + j - 1;
                if (i > 0) {
                    f[i][j] |= (f[i - 1][j] && s1[i - 1] == s3[p]);
                }
                if (j > 0) {
                    f[i][j] |= (f[i][j - 1] && s2[j - 1] == s3[p]);
                }
            }
        }

        return f[n][m];
    }
};
```

因为这里数组 $f$ 的第 $i$ 行只和第 $i - 1$ 行相关，所以我们可以用**滚动数组优化**这个动态规划，这样空间复杂度可以变成 $O(m)$。

``` c++
class Solution {
public:
    bool isInterleave(string s1, string s2, string s3) {
        auto f = vector <int> (s2.size() + 1, false);

        int n = s1.size(), m = s2.size(), t = s3.size();

        if (n + m != t) {
            return false;
        }

        f[0] = true;
        for (int i = 0; i <= n; ++i) {
            for (int j = 0; j <= m; ++j) {
                int p = i + j - 1;
                if (i > 0) {
                    f[j] &= (s1[i - 1] == s3[p]);
                }
                if (j > 0) {
                    f[j] |= (f[j - 1] && s2[j - 1] == s3[p]);
                }
            }
        }

        return f[m];
    }
};
```

## 验证二叉搜索树

### 递归

我们设计一个递归函数 `helper(root, lower, upper)​` 来递归判断，函数表示考虑以 $root$ 为根的子树，判断子树中所有节点的值是否都在 $(l,r)$ 的范围内（注意是**开区间**）。如果 $root$ 节点的值 $val$ 不在 $(l,r)$ 的范围内说明不满足条件直接返回，否则我们要继续递归调用检查它的左右子树是否满足，如果都满足才说明这是一棵二叉搜索树。

那么根据二叉搜索树的性质，在递归调用左子树时，我们需要把上界 $upper$ 改为 $root.val$，即调用 `helper(root.left, lower, root.val)`，因为左子树里所有节点的值均小于它的根节点的值。同理递归调用右子树时，我们需要把下界 $lower$ 改为 $root.val$，即调用 `helper(root.right, root.val, upper)`。

函数递归调用的入口为 `helper(root, -inf, +inf)`， `inf` 表示一个无穷大的值。

``` c++
class Solution {
public:
    bool helper(TreeNode* root, long long lower, long long upper) {
        if (root == nullptr) {
            return true;
        }
        if (root -> val <= lower || root -> val >= upper) {
            return false;
        }
        return helper(root -> left, lower, root -> val) && helper(root -> right, root -> val, upper);
    }
    bool isValidBST(TreeNode* root) {
        return helper(root, LONG_MIN, LONG_MAX);
    }
};
```

### 中序遍历

基于方法一中提及的性质，我们可以进一步知道二叉搜索树「中序遍历」得到的值构成的序列一定是升序的，这启示我们在中序遍历的时候实时检查当前节点的值是否大于前一个中序遍历到的节点的值即可。如果均大于说明这个序列是升序的，整棵树是二叉搜索树，否则不是。

``` c++
class Solution {
public:
    bool isValidBST(TreeNode* root) {
        stack<TreeNode*> stack;
        long long inorder = (long long)INT_MIN - 1;

        while (!stack.empty() || root != nullptr) {
            while (root != nullptr) {
                stack.push(root);
                root = root -> left;
            }
            root = stack.top();
            stack.pop();
            // 如果中序遍历得到的节点的值小于等于前一个 inorder，说明不是二叉搜索树
            if (root -> val <= inorder) {
                return false;
            }
            inorder = root -> val;
            root = root -> right;
        }
        return true;
    }
};
```

## 恢复二叉搜索树

### 显式中序遍历

我们需要考虑两个节点被错误地交换后对原二叉搜索树造成了什么影响。对于二叉搜索树，我们知道如果对其进行中序遍历，得到的值序列是递增有序的，而如果我们错误地交换了两个节点，等价于在这个值序列中交换了两个值，破坏了值序列的递增性。解题算法步骤如下：

1. 找到二叉搜索树中序遍历得到值序列的不满足条件的位置。
2. 如果有两个，我们记为 $i$ 和 $j$ （$i<j$ 且 $a_i>a_{i+1}~\&\&~ a_j>a_{j+1}$），那么对应被错误交换的节点即为 $a_i$ 对应的节点和 $a_{j+1}$ 对应的节点，我们分别记为 $x$ 和 $y$。
3. 如果有一个，我们记为 $i$，那么对应被错误交换的节点即为 $a_i$ 对应的节点和 $a_{i+1}$ 对应的节点，我们分别记为 $x$ 和 $y$。
4. 交换 $x$ 和 $y$ 两个节点即可。

实现中，开辟一个新数组 \textit{nums}nums 来记录中序遍历得到的值序列，然后线性遍历找到两个位置 ii 和 jj，并重新遍历原二叉搜索树修改对应节点的值完成修复。

``` c++
class Solution {
public:
    void inorder(TreeNode* root, vector<int>& nums) {
        if (root == nullptr) {
            return;
        }
        inorder(root->left, nums);
        nums.push_back(root->val);
        inorder(root->right, nums);
    }

    pair<int,int> findTwoSwapped(vector<int>& nums) {
        int n = nums.size();
        int x = -1, y = -1;
        for(int i = 0; i < n - 1; ++i) {
            if (nums[i + 1] < nums[i]) {
                y = nums[i + 1];
                if (x == -1) {
                    x = nums[i];
                }
                else break;
            }
        }
        return {x, y};
    }
    
    void recover(TreeNode* r, int count, int x, int y) {
        if (r != nullptr) {
            if (r->val == x || r->val == y) {
                r->val = r->val == x ? y : x;
                if (--count == 0) {
                    return;
                }
            }
            recover(r->left, count, x, y);
            recover(r->right, count, x, y);
        }
    }

    void recoverTree(TreeNode* root) {
        vector<int> nums;
        inorder(root, nums);
        pair<int,int> swapped= findTwoSwapped(nums);
        recover(root, 2, swapped.first, swapped.second);
    }
};
```

### 隐式中序遍历

方法一是显式地将中序遍历的值序列保存在一个 $\textit{nums}$ 数组中，然后再去寻找被错误交换的节点，但我们也可以隐式地在中序遍历的过程就找到被错误交换的节点 $x$ 和 $y$。

具体来说，由于我们只关心中序遍历的值序列中每个**相邻的位置的大小关系是否满足条件**，且错误交换后**最多两个位置不满足条件**，因此在中序遍历的过程我们只需要维护当前中序遍历到的最后一个节点 $\textit{pred}$，然后在遍历到下一个节点的时候，看两个节点的值是否满足前者小于后者即可，如果不满足说明找到了一个交换的节点，且在找到两次以后就可以终止遍历。

这样我们就可以在中序遍历中直接找到被错误交换的两个节点 $x$ 和 $y$，不用显式建立 $\textit{nums}$ 数组。

``` c++
class Solution {
public:
    void recoverTree(TreeNode* root) {
        stack<TreeNode*> stk;
        TreeNode* x = nullptr;
        TreeNode* y = nullptr;
        TreeNode* pred = nullptr;

        while (!stk.empty() || root != nullptr) {
            while (root != nullptr) {
                stk.push(root);
                root = root->left;
            }
            root = stk.top();
            stk.pop();
            if (pred != nullptr && root->val < pred->val) {
                y = root;
                if (x == nullptr) {
                    x = pred;
                }
                else break;
            }
            pred = root;
            root = root->right;
        }

        swap(x->val, y->val);
    }
};
```

### Morris中序遍历

**Morris遍历算法**能将非递归的中序遍历的空间复杂度将为 $O(1)$，整体步骤如下：

1. 如果 $x$ 无左孩子，则访问 $x$ 的右孩子，即 $x=x.right$
2. 如果 $x$ 有左孩子，则找到 $x$ 左子树上最右的节点（**即左子树中序遍历的最后一个节点，$x$ 在中序遍历中的前驱节点**），我们记为 $predecessor$。根据 $predecessor$ 的右孩子是否为空，进行如下操作：
   - 如果 $predecessor$ 的右孩子为空，则将其右孩子指向 $x$，然后访问 $x$ 的左孩子，即 $x = x.\textit{left}$。
   - 如果 $\textit{predecessor}$ 的右孩子不为空，则此时其右孩子指向 $x$，说明我们已经遍历完 $x$ 的左子树，我们将 $\textit{predecessor}$ 的右孩子置空，然后访问 $x$ 的右孩子，即 $x = x.\textit{right}$。

3. 重复上述操作，直至访问玩整棵树

```  c++
class Solution {
public:
    void recoverTree(TreeNode* root) {
        TreeNode *x = nullptr, *y = nullptr, *pred = nullptr, *predecessor = nullptr;

        while (root != nullptr) {
            if (root->left != nullptr) {
                // predecessor 节点就是当前 root 节点向左走一步，然后一直向右走至无法走为止
                predecessor = root->left;
                while (predecessor->right != nullptr && predecessor->right != root) {
                    predecessor = predecessor->right;
                }
                
                // 让 predecessor 的右指针指向 root，继续遍历左子树
                if (predecessor->right == nullptr) {
                    predecessor->right = root;
                    root = root->left;
                }
                // 说明左子树已经访问完了，我们需要断开链接
                else {
                    if (pred != nullptr && root->val < pred->val) {
                        y = root;
                        if (x == nullptr) {
                            x = pred;
                        }
                    }
                    pred = root;

                    predecessor->right = nullptr;
                    root = root->right;
                }
            }
            // 如果没有左孩子，则直接访问右孩子
            else {
                if (pred != nullptr && root->val < pred->val) {
                    y = root;
                    if (x == nullptr) {
                        x = pred;
                    }
                }
                pred = root;
                root = root->right;
            }
        }
        swap(x->val, y->val);
    }
};
```

## 相同的树

### 深度优先搜索

如果两个二叉树都为空，则两个二叉树相同。如果两个二叉树中有且只有一个为空，则两个二叉树一定不相同。

如果两个二叉树都不为空，那么首先判断它们的根节点的值是否相同，若不相同则两个二叉树一定不同，若相同，再分别判断两个二叉树的左子树是否相同以及右子树是否相同。这是一个递归的过程，因此可以使用深度优先搜索，递归地判断两个二叉树是否相同。

``` c++
class Solution {
public:
    bool isSameTree(TreeNode* p, TreeNode* q) {
        if (p == nullptr && q == nullptr) {
            return true;
        } else if (p == nullptr || q == nullptr) {
            return false;
        } else if (p->val != q->val) {
            return false;
        } else {
            return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
        }
    }
};
```

### 广度优先搜索

使用两个队列分别存储两个二叉树的节点。初始时将两个二叉树的根节点分别加入两个队列。每次从两个队列各取出一个节点，进行如下比较操作。

1. 比较两个节点的值，如果两个节点的值不相同则两个二叉树一定不同；
2. 如果两个节点的值相同，则判断两个节点的子节点是否为空，如果只有一个节点的左子节点为空，或者只有一个节点的右子节点为空，则两个二叉树的结构不同，因此两个二叉树一定不同；
3. 如果两个节点的子节点的结构相同，则将两个节点的非空子节点分别加入两个队列，子节点加入队列时需要注意顺序，如果左右子节点都不为空，则先加入左子节点，后加入右子节点。

如果搜索结束时两个队列同时为空，则两个二叉树相同。如果只有一个队列为空，则两个二叉树的结构不同，因此两个二叉树不同。

``` c++
class Solution {
public:
    bool isSameTree(TreeNode* p, TreeNode* q) {
        if (p == nullptr && q == nullptr) {
            return true;
        } else if (p == nullptr || q == nullptr) {
            return false;
        }
        queue <TreeNode*> queue1, queue2;
        queue1.push(p);
        queue2.push(q);
        while (!queue1.empty() && !queue2.empty()) {
            auto node1 = queue1.front();
            queue1.pop();
            auto node2 = queue2.front();
            queue2.pop();
            if (node1->val != node2->val) {
                return false;
            }
            auto left1 = node1->left, right1 = node1->right, left2 = node2->left, right2 = node2->right;
            if ((left1 == nullptr) ^ (left2 == nullptr)) {
                return false;
            }
            if ((right1 == nullptr) ^ (right2 == nullptr)) {
                return false;
            }
            if (left1 != nullptr) {
                queue1.push(left1);
            }
            if (right1 != nullptr) {
                queue1.push(right1);
            }
            if (left2 != nullptr) {
                queue2.push(left2);
            }
            if (right2 != nullptr) {
                queue2.push(right2);
            }
        }
        return queue1.empty() && queue2.empty();
    }
};
```





