LeetCode刷题

 [TOC]
## 二叉树的中序遍历
### Morris遍历算法（假设当前遍历到的节点是x）
- 如果x无左孩子，先将x的值加入到答案数组，再访问x的右孩子，即```x=x.right```
- 如果x有左孩子，则找到x左子树上最右的节点，记为predecessor。根据predecessor的右孩子是否为空，采用下列操作：
    + 如果predecessor的右孩子为空，则将predecessor的右孩子指向x，然后再访问x的左子树，即```x=x.left```
    + 如果predecessor的右孩子不为空，则此时其右孩子指向x，说明我们已经遍历完x的左子树，我们将predecessor的右孩子置空，将x的值加入
    到答案数组，然后访问x的右子树，即```x=x.right```
    <img src="https://media.giphy.com/media/FvT0B3e71HR9Cp4IAm/giphy.gif" width="1000" height="500"   >

```c++
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> res;
        TreeNode *predecessor = nullptr;

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
                    res.push_bac*k*(root->val);
                    predecessor->right = nullptr;
                    root = root->right;
                }
            }
            // 如果没有左孩子，则直接访问右孩子
            else {
                res.push_bac*k*(root->val);
                root = root->right;
            }
        }
        return res;
    }
};
```
## 寻找两个有序数组的中位数
### 最直观的思路
- 使用归并的方式，合并两个有序数组，得到一个大的有序数组。大的有序数组的中间位置的元素，即为中位数
- 不需要合并两个有序数组，只要找到中位数的位置即可。由于两个数组的长度已知，因此中位数对应的两个数组的下标之和也是已知的。维护两个指针，初始时分别指向两个数组的下标 0 的位置，每次将指向较小值的指针后移一位（如果一个指针已经到达数组末尾，则只需要移动另一个数组的指针），直到到达中位数的位置。
以上两种方法的时间复杂度都为 $O(m+n)$，空间复杂度分别为 $O(m+n)$和 $O(1)$，不满足题目要求的时间复杂度 $O(log(m+n))$
### 二分查找
如果对时间复杂度的要求有 $log$，通常都需要用到```二分查找```。
这道题可以转化成寻找两个有序数组中的第*k*小的数，其中 $k=(m+n)/2$ 或 $k=(m+n)/2+1$
假设两个有序数组分别是A和B。要找到第 $k$ 个元素，可以比较 $A[k/2-1]$ 和 $B[k/2-1]$，对于 $A[k/2-1]$ 和 $B[k/2-1]$ 中的较小值，最多只会有 $(k/2-1)+(k/2-1)<=k-2$ 个元素比它小。由此可以归纳出两种情况：

- 如果 $A[k/2-1]<=B[k/2-1]$，$A[0]$ 到 $A[k/2-1]$ 都不可能第 $k$ 个数，可以全部排除
- 如果 $A[k/2-1]>B[k/2-1]$，$B[0]$ 到 $B[k/2-1]$ 都不可能是第 $k$ 个数，可以全部排除

可以看到，比较A[*k*/2-1]和B[*k*/2-1]之后，可以排除*k*/2个不可能是第*k*小的数，查找范围缩小了一半。同时，我们将在排除后的新数组上继续进行二分查找，并且根据我们排除数的个数，减少 *k* 的值，这是因为我们排除的数都不大于第 *k* 小的数。
有以下三种情况需要特殊处理：

- 如果A[*k*/2-1]或者B[*k*/2-1]越界，那么可以选取对应数组中的最后一个元素。在这种情况下，**必须根据排除数分个数减少*k*的值**， 而不能直接将*k*-*k*/2
- 如果一个数组为空，说明该数组中的所有元素都被排除，我们可以直接返回另一个数组中第*k*小的元素
- 如果*k*=1，只需返回两个数组首元素的最小值即可

``` c++
class Solution {
public:
    int getKthElement(const vector<int>& nums1, const vector<int>& nums2, int k) {
        /* 主要思路：要找到第 k (k>1) 小的元素，那么就取 pivot1 = nums1[k/2-1] 和 pivot2 = nums2[k/2-1] 进行比较
         * 这里的 "/" 表示整除
         * nums1 中小于等于 pivot1 的元素有 nums1[0 .. k/2-2] 共计 k/2-1 个
         * nums2 中小于等于 pivot2 的元素有 nums2[0 .. k/2-2] 共计 k/2-1 个
         * 取 pivot = min(pivot1, pivot2)，两个数组中小于等于 pivot 的元素共计不会超过 (k/2-1) + (k/2-1) <= k-2 个
         * 这样 pivot 本身最大也只能是第 k-1 小的元素
         * 如果 pivot = pivot1，那么 nums1[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums1 数组
         * 如果 pivot = pivot2，那么 nums2[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums2 数组
         * 由于我们 "删除" 了一些元素（这些元素都比第 k 小的元素要小），因此需要修改 k 的值，减去删除的数的个数
         */

        int m = nums1.size();
        int n = nums2.size();
        int index1 = 0, index2 = 0;

        while (true) {
            // 边界情况
            if (index1 == m) {
                return nums2[index2 + k - 1];
            }
            if (index2 == n) {
                return nums1[index1 + k - 1];
            }
            if (k == 1) {
                return min(nums1[index1], nums2[index2]);
            }

            // 正常情况
            int newIndex1 = min(index1 + k / 2 - 1, m - 1);
            int newIndex2 = min(index2 + k / 2 - 1, n - 1);
            int pivot1 = nums1[newIndex1];
            int pivot2 = nums2[newIndex2];
            if (pivot1 <= pivot2) {
                k -= newIndex1 - index1 + 1;
                index1 = newIndex1 + 1;
            }
            else {
                k -= newIndex2 - index2 + 1;
                index2 = newIndex2 + 1;
            }
        }
    }

    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int totalLength = nums1.size() + nums2.size();
        if (totalLength % 2 == 1) {
            return getKthElement(nums1, nums2, (totalLength + 1) / 2);
        }
        else {
            return (getKthElement(nums1, nums2, totalLength / 2) + getKthElement(nums1, nums2, totalLength / 2 + 1)) / 2.0;
        }
    }
};
```
### 划分数组
- 中位数的作用是什么：
> 将一个集合划分为两个长度相等的子集，其中一个子集的元素总是大于另一个子集中的元素

- 首先，在任意位置*i*将A划分为两个部分：
```
           left_A            |          right_A
    A[0], A[1], ..., A[i-1]  |  A[i], A[i+1], ..., A[m-1]
```
由于A有*m*个元素，所以有*m*+1中划分的方法
> len(left_A) = *i*, len(right_A) = *m-i*

- 采用同样的方式，在任意位置*j*将B划分为两个部分：
```
           left_B            |          right_B
    B[0], B[1], ..., B[j-1]  |  B[j], B[j+1], ..., B[n-1]
```
- 将两组划分对应合并：
```
          left_part          |         right_part
    A[0], A[1], ..., A[i-1]  |  A[i], A[i+1], ..., A[m-1]
    B[0], B[1], ..., B[j-1]  |  B[j], B[j+1], ..., B[n-1]
```
- 当A和B的总长度是偶数时，如果可以确认：
   + len(left_part) = len(right_part)
   + max(left_part) <= min(right_part)

  此时，中位数就是前一部分的最大值和后一部分的最小值的平均值
- 当A和B的总长度是奇数时，如果可以确认：
   + len(left_part) = len(right_part) + 1
   + max(left_part) <= min(right_part)

  此时，中位数就是前一部分的最大值
- 第一个条件对于总长度是偶数和奇数的情况有所不同，但是可以将两种情况合并。第二个条件对于总长度是偶数和奇数的情况是一样的。要确保这两个条件，需要保证：
   + *i + j = m - i + n - j*（当*m + n*为偶数）或*i + j = m - i + n - j + 1*（当*m + n*为奇数）。
   + B[*j*-1] <= A[*i*] 以及A[*i*-1] <= B[*j*]

- 我们需要做的是：
> 在[0，*m*]中找到*i*，使得
>  B[*j*-1] <= A[*i*] 且A[*i*-1] <= B[*j*]， 其中*j* = $\frac {m+n+1}{2}$ - *i*

  可以证明它等价于：
  > 在[0，*m*]中找到最大的*i*，使得
  >  A[*i*-1] <= B[*j*]， 其中*j* = $\frac {m+n+1}{2}$ - *i*

  这是因为：
  + 当 *i* 从 0 ∼ *m* 递增时，A[*i*-1] 递增，B[*j*] 递减，所以一定存在一个最大的 *i* 满足A[*i*-1] <= B[*j*]；
  + 如果*i*是最大的，那么说明*i*+1不满足，有A[*i*]>B[*j*-1]

``` c++ 
class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        if (nums1.size() > nums2.size()) {
            return findMedianSortedArrays(nums2, nums1);
        }
        
        int m = nums1.size();
        int n = nums2.size();
        int left = 0, right = m, ansi = -1;
        // median1：前一部分的最大值
        // median2：后一部分的最小值
        int median1 = 0, median2 = 0;

        while (left <= right) {
            // 前一部分包含 nums1[0 .. i-1] 和 nums2[0 .. j-1]
            // 后一部分包含 nums1[i .. m-1] 和 nums2[j .. n-1]
            int i = (left + right) / 2;
            int j = (m + n + 1) / 2 - i;

            // nums_im1, nums_i, nums_jm1, nums_j 分别表示 nums1[i-1], nums1[i], nums2[j-1], nums2[j]
            int nums_im1 = (i == 0 ? INT_MIN : nums1[i - 1]);
            int nums_i = (i == m ? INT_MAX : nums1[i]);
            int nums_jm1 = (j == 0 ? INT_MIN : nums2[j - 1]);
            int nums_j = (j == n ? INT_MAX : nums2[j]);

            if (nums_im1 <= nums_j) {
                ansi = i;
                median1 = max(nums_im1, nums_jm1);
                median2 = min(nums_i, nums_j);
                left = i + 1;
            }
            else {
                right = i - 1;
            }
        }

        return (m + n) % 2 == 0 ? (median1 + median2) / 2.0 : median1;
    }
};
```
## 最长回文子串
### 动态规划
对于一个子串而言，如果它是回文串，并且长度大于 2，那么将它首尾的两个字母去除之后，它仍然是个回文串。由此想到可以用`动态规划`解该问题。
用$P(i,j)$表示字符串 *s* 的第 *i* 到 *j* 个字母组成的串是否为回文串：
$$
P(i, j) = \begin{cases} true & 子串S_i...S_j是回文串 \\ false & 其它情况 \end{cases}
$$
由此可写出动态规划的状态转移方程：
$$
P(i, j) = P(i+1, j-1) \bigwedge S_i == S_j
$$
上文的所有讨论是建立在子串长度大于 2 的前提之上的，我们还需要考虑动态规划中的边界条件，即子串的长度为 1 或 2。对于长度为 1 的子串，它显然是个回文串；对于长度为 2 的子串，只要它的两个字母相同，它就是一个回文串。因此我们就可以写出动态规划的边界条件：
$$
\begin{cases} P(i, i) = true \\ P(i, i+1) = (S_i == S_{i+1}) \end{cases}
$$
最终的答案即为所有 $P(i,j)=true$ 中 $j-i+1$（即子串长度）的最大值。
``` c++
class Solution {
public:
    string longestPalindrome(string s) {
        int n = s.size();
        vector<vector<int>> dp(n, vector<int>(n));
        string ans;
        for (int l = 0; l < n; ++l) {
            for (int i = 0; i + l < n; ++i) {
                int j = i + l;
                if (l == 0) {
                    dp[i][j] = 1;
                } else if (l == 1) {
                    dp[i][j] = (s[i] == s[j]);
                } else {
                    dp[i][j] = (s[i] == s[j] && dp[i + 1][j - 1]);
                }
                if (dp[i][j] && l + 1 > ans.size()) {
                    ans = s.substr(i, l + 1);
                }
            }
        }
        return ans;
    }
};
```
### 中心扩散
根据动态规划中的状态转移方程，我们可以找出其中的状态转移链：
$$
P(i,j) \leftarrow P(i+1, j-1) \leftarrow P(i+2, j-2) \leftarrow ... \leftarrow 某一边界情况
$$
**边界情况**即为子串长度为 1 或 2 的情况，也叫作`回文中心`。我们枚举每一种边界情况，并从对应的子串开始不断地向两边扩展。
该方法的本质是：枚举所有的`回文中心`并尝试扩展，直到无法扩展为止，此时的回文串长度即为此`回文中心`下的最长回文串长度。我们对所有的长度求出最大值，即可得到最终的答案。`
``` c++
class Solution {
    public String longestPalindrome(String s) {
        if (s == null || s.length() < 1) {
            return "";
        }
        int start = 0, end = 0;
        for (int i = 0; i < s.length(); i++) {
            int len1 = expandAroundCenter(s, i, i);
            int len2 = expandAroundCenter(s, i, i + 1);
            int len = Math.max(len1, len2);
            if (len > end - start) {
                start = i - (len - 1) / 2;
                end = i + len / 2;
            }
        }
        return s.substring(start, end + 1);
    }

    public int expandAroundCenter(String s, int left, int right) {
        while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            --left;
            ++right;
        }
        return right - left - 1;
    }
}
```

## Z字形变换
### 按行排序
- 使用`min(numRows, len(s))`个列表来表示Z字形图案中的非空行
- 从左至右迭代*s*，使用当前行和当前方向这两个变量对合适的行进行跟踪
- 只有当我们向上移动到最上面的行或向下移动到最下面的行时，当前方向才会发生改变
``` c++
class Solution {
public:
    string convert(string s, int numRows) {

        if (numRows == 1) return s;

        vector<string> rows(min(numRows, int(s.size())));
        int curRow = 0;
        bool goingDown = false;

        for (char c : s) {
            rows[curRow] += c;
            if (curRow == 0 || curRow == numRows - 1) goingDown = !goingDown;
            curRow += goingDown ? 1 : -1;
        }

        string ret;
        for (string row : rows) ret += row;
        return ret;
    }
};
```
### 按行访问
- 首先访问`行0`的所有元素，然后`行1`、`行2`的，以此类推
- 对于所有整数*k*有：
    + 行`0`中的字符位于索引$k*(2*numRows - 2)$处
    + 行`numRows-1`的字符位于索引$ numRows-1 + k*(2*numRows - 2) $处
    + 行`i`中的字符位于索引$ i + k*(2*numRows - 2) $和$  (k+1)*(2*numRows - 2) - i $处
``` c++
class Solution {
public:
    string convert(string s, int numRows) {

        if (numRows == 1) return s;

        string ret;
        int n = s.size();
        int cycleLen = 2 * numRows - 2;

        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j + i < n; j += cycleLen) {
                ret += s[j + i];
                if (i != 0 && i != numRows - 1 && j + cycleLen - i < n)
                    ret += s[j + cycleLen - i];
            }
        }
        return ret;
    }
};
```
## 整数反转
重复“弹出” x 的最后一位数字，并将它“推入”到 rev 的后面:
```
//pop operation:
pop = x % 10;
x /= 10;

//push operation:
temp = rev * 10 + pop;
rev = temp;
```
`注意`当$temp = rev*10 + pop$时会导致溢出，假设rev是正数：

1. 如果$temp = rev*10 + pop$导致溢出，那么一定有$rev > \frac{INIT\_MAX}{10}$
2. 如果$rev == \frac{INIT\_MAX}{10}$， 那么只要$pop > 7$，$temp = rev*10 + pop$就会溢出
![avatar](https://pic.leetcode-cn.com/42c736510f4914af169907d61b22d1a39bd5a16bbd7eca0466d90350e2763164-2.jpg)
当 rev 为负时，逻辑相似。
![avatar](https://pic.leetcode-cn.com/525aa75c19702e57b780c91a7ebb990359b14e96acc09b6327d9e1f0a5b3a16a-3.jpg)
``` c++
class Solution {
public:
    int reverse(int x) {
        int rev = 0;
        while (x != 0) {
            int pop = x % 10;
            x /= 10;
            if (rev > INT_MAX/10 || (rev == INT_MAX / 10 && pop > 7)) return 0;
            if (rev < INT_MIN/10 || (rev == INT_MIN / 10 && pop < -8)) return 0;
            rev = rev * 10 + pop;
        }
        return rev;
    }
};
```
- 时间复杂度分析：$\lg(x)$，x 中大约有$\lg(x)$位数字
## 字符串转换为整数(Atoi)
对于溢出的处理方式通常可以转换为 *INT_MAX* 的逆操作。比如判断某数乘 10 是否会溢出，那么就把该数和 *INT_MAX* 除 10 进行比较
### 自动机
程序在每个时刻有一个状态 `s`，每次从序列中输入一个字符 `c`，并根据字符 `c` 转移到下一个状态 `s'`。这样，我们只需要建立一个覆盖所有情况的从 `s` 与 `c` 映射到 `s'` 的表格即可解决题目中的问题。表格如下：

''	+/-	number	other
start	start	signed	in_number	end
signed	end	end	in_number	end
in_number	end	end	in_number	end
end	end	end	end	end
也可以用下图更直观地展示：
<img src="https://assets.leetcode-cn.com/solution-static/8_fig1.PNG" width="600" height="400" align="center">
``` c++
class Automaton {
    string state = "start";
    unordered_map<string, vector<string>> table = {
        {"start", {"start", "signed", "in_number", "end"}},
        {"signed", {"end", "end", "in_number", "end"}},
        {"in_number", {"end", "end", "in_number", "end"}},
        {"end", {"end", "end", "end", "end"}}
    };

    int get_col(char c) {
        if (isspace(c)) return 0;
        if (c == '+' or c == '-') return 1;
        if (isdigit(c)) return 2;
        return 3;
    }
public:
    int sign = 1;
    long long ans = 0;

    void get(char c) {
        state = table[state][get_col(c)];
        if (state == "in_number") {
            ans = ans * 10 + c - '0';
            ans = sign == 1 ? min(ans, (long long)INT_MAX) : min(ans, -(long long)INT_MIN);
        }
        else if (state == "signed")
            sign = c == '+' ? 1 : -1;
    }
};

class Solution {
public:
    int myAtoi(string str) {
        Automaton automaton;
        for (char c : str)
            automaton.get(c);
        return automaton.sign * automaton.ans;
    }
};
```
## 正则表达式匹配
### 动态规划思想
题目中的匹配是一个「逐步匹配」的过程：我们每次从字符串 *p* 中取出一个字符或者「字符 + 星号」的组合，并在 *s* 中进行匹配。对于 *p* 中一个字符而言，它只能在 *s* 中匹配一个字符，匹配的方法具有唯一性；而对于 *p* 中字符 + 星号的组合而言，它可以在 *s* 中匹配任意自然数个字符，并不具有唯一性。因此我们可以考虑使用`动态规划`，对匹配的方案进行枚举。
用 $f[i][j]$ 表示 *s* 的前 *i* 个字符与 *p* 中的前 *j* 个字符是否能够匹配。在进行状态转移时，考虑 *p* 的第 *j* 个字符的匹配情况：

- $p$ 中的第 $j$ 个字符是一个`小写字母 `，则有：
$$
f[i][j] = \begin{cases} f[i-1][j-1], & s[i] == p[j] \\ false, & s[i] \neq p[j] \end{cases}
$$
- 如果 $p$ 的第 $j$ 个字符是 `*`，那么就表示我们可以对 $p$ 的第 $j-1$ 个字符匹配任意自然数次。这种**字母+星号**的组合在匹配的过程中，本质上只会有两种情况：
    + 匹配 $s$ 末尾的一个字符，将该字符扔掉，而该组合还可以继续匹配；
    + 不匹配字符，将该组合扔掉，不再进行匹配。
    根据这一本质，可以写出状态转移方程：
$$
f[i][j] = \begin{cases} f[i-1][j] or f[i][j-2], & s[i] == p[j-1] \\ f[i][j-2], & s[i] \neq p[j-2] \end{cases}
$$
- 在任意情况下，只要 $p[j]$ 是`.`，那么 $p[j]$ 一定匹配 $s$ 中的任意一个小写字母。
最终的状态转移方程如下：
$$
f[i][j] = \begin{cases} if (p[j] \neq '*') &= \begin{cases} f[i-1][j-1], & match(s[i], p[j]) \\ false, & otherwise \end{cases} \\ otherwise &= \begin{cases}  f[i-1][j] or f[i][j-2], & match(s[i], p[j-1]) \\ f[i][j-2], & otherwise \end{cases} \end{cases}
$$
### 细节
动态规划的边界条件为 $f[0][0] = true$，即两个空字符串是可以匹配的。最终的答案即为 *f[m][n]*，其中 *m* 和 *n* 分别是字符串 *s* 和 *p* 的长度。
``` c++
class Solution {
public:
    bool isMatch(string s, string p) {
        int m = s.size();
        int n = p.size();

        auto matches = [&](int i, int j) {
            if (i == 0) {
                return false;
            }
            if (p[j - 1] == '.') {
                return true;
            }
            return s[i - 1] == p[j - 1];
        };

        vector<vector<int>> f(m + 1, vector<int>(n + 1));
        f[0][0] = true;
        for (int i = 0; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (p[j - 1] == '*') {
                    f[i][j] |= f[i][j - 2];
                    if (matches(i, j - 1)) {
                        f[i][j] |= f[i - 1][j];
                    }
                }
                else {
                    if (matches(i, j)) {
                        f[i][j] |= f[i - 1][j - 1];
                    }
                }
            }
        }
        return f[m][n];
    }
};
```
### 注意 
## 盛最多水的容器
### 双指针
在初始时，左右指针分别指向数组的左右两端，容纳的水量是：
$$
两个指针指向的数字中较小值*指针之间的距离
$$
然后移动数字较小的那个指针，直到左右指针相遇。
### 证明
> 双指针代表了什么？

双指针代表的是 **可以作为容器边界的所有位置的范围**。在一开始，双指针指向数组的左右边界，表示 **数组中所有的位置都可以作为容器的边界**。之后，我们每次将 **对应的数字较小的那个指针**
往 **另一个指针** 的方向移动一个位置，就表示我们认为 **这个指针不可能再作为容器的边界了**。
> 为什么对应的数字较小的那个指针不可能再作为容器的边界了？

初始时，假设左指针与右指针指向的数分别为 *x* 和 *y*，不失一般性， 我们假设 $x \leqslant y$。同时，两指针间的距离为 *t*。那么，它们组成的容器的容量为：
$$
min(x,y)*t == x*t
$$
我们可以断定，**如果我们保持左指针的位置不变，那么无论右指针在哪里，这个容器的容量都不会超过 $x * t$ 了**。
任意向左移动右指针，假设指向的数为 $y_1$， 两个指针间的距离为 $t_1$，那么显然有 $t_1 < t$，并且 $min(x, y_1) \leqslant min(x, y)$。
因此有：
$$
min(x, y_1)*t < min(x, y)*t
$$
即无论如何移动右指针，得到的容器的容量都小于移动前容器的容量。也就是说，**这个左指针对应的数不会作为容器的边界了**，那么就可以丢弃这个位置，**将左指针向右移动一个位置**。
思考问题的方式变为：
- 求出当前左右指针对应容器的容量；
- 对应数字较小的那个指针以后不可能作为容器的边界了，将其丢弃，并移动对应的指针。
``` c++
class Solution {
public:
    int maxArea(vector<int>& height) {
        int l = 0, r = height.size() - 1;
        int ans = 0;
        while (l < r) {
            int area = min(height[l], height[r]) * (r - l);
            ans = max(ans, area);
            if (height[l] <= height[r]) {
                ++l;
            }
            else {
                --r;
            }
        }
        return ans;
    }
};
```
## 整数转罗马数字
### 打表法
``` c++
class Solution {
public:
    string intToRoman(int num) {
        // 打表法
        vector< vector<string>> tb = {{"I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"},
                                  {"X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"},
                                  {"C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"},
                                  {"M", "MM", "MMM"}};
    
        string ans = "";
        int carry = 0;
        while(num != 0) {
            int p = num % 10;
            if (p == 0) {
                num /= 10;
                ++carry;
                continue;
            }
            string tmp = tb[carry][p-1] + ans;
            ans = tmp;
            num /= 10;
            ++carry;
        }

        return ans;
    }
};
```
### 贪心法
将给定的整数转换为罗马数字需要找到下面图中 13 个符号的序列，这些符号的对应值加起来就是整数。根据符号值，此序列必须按从大到小的顺序排列。
![avatar](https://img-blog.csdnimg.cn/20200414105909472.png)
为了表示一个给定的整数，我们寻找适合它的最大符号。我们减去它，然后寻找适合余数的最大符号，依此类推，直到余数为0。我们取出的每个符号都附加到输出的罗马数字字符串上。
``` c++
class Solution {
public:
    string intToRoman(int num) {
        // 贪心法
        vector< pair<int, string>> tb = {{1000, "M"}, {900, "CM"}, {500, "D"}, {400, "CD"},
                                        {100, "C"}, {90, "XC"}, {50, "L"}, {40, "XL"},
                                        {10, "X"}, {9, "IX"}, {5, "V"}, {4, "IV"}, {1, "I"}};

        string ans = "";
        int last = 0;
        while(num != 0) {
            for(int i = last; i < tb.size(); ++i) {
                if (tb[i].first <= num) {
                    ans += tb[i].second;
                    num -= tb[i].first;
                    last = i;
                    break;
                }
            }
        }

        return ans;
    }
};
```
## 最长公共前缀(Longest Common Prefix, LCP)
### 横向扫描
用$LCP(S_1,...,S_n)$表示字符串 $S_1,...,S_n$ 的最长公共前缀。可以得到以下结论：
$$
LCP(S_1,...,S_n)=LCP(LCP(LCP(S_1, S_2), S_3),...,S_n)
$$
依次遍历字符串数组中的每个字符串，对于每个遍历到的字符串，更新最长公共前缀，当遍历完所有的字符串以后，即可得到字符串数组中的最长公共前缀。如果在尚未遍历完所有的字符串时，最长公共前缀已经是空串，则最长公共前缀一定是空串，因此不需要继续遍历剩下的字符串，直接返回空串即可。示意图如下：
<img src="https://assets.leetcode-cn.com/solution-static/14/14_fig1.png" width="600" height="400" align="center">
``` c++
class Solution {
public:
    string longestCommonPrefix(vector<string>& strs) {
        if (!strs.size()) {
            return "";
        }
        string prefix = strs[0];
        int count = strs.size();
        for (int i = 1; i < count; ++i) {
            prefix = longestCommonPrefix(prefix, strs[i]);
            if (!prefix.size()) {
                break;
            }
        }
        return prefix;
    }

    string longestCommonPrefix(const string& str1, const string& str2) {
        int length = min(str1.size(), str2.size());
        int index = 0;
        while (index < length && str1[index] == str2[index]) {
            ++index;
        }
        return str1.substr(0, index);
    }
};
```
### 纵向扫描
从前往后遍历所有字符串的每一列，比较相同列上的字符是否相同，如果相同则继续对下一列进行比较，如果不相同则当前列不再属于公共前缀，当前列之前的部分为最长公共前缀。示意图如下：
<img src="https://assets.leetcode-cn.com/solution-static/14/14_fig2.png" width="600" height="400" align="center">
``` c++
class Solution {
public:
    string longestCommonPrefix(vector<string>& strs) {
        if (!strs.size()) {
            return "";
        }
        int length = strs[0].size();
        int count = strs.size();
        for (int i = 0; i < length; ++i) {
            char c = strs[0][i];
            for (int j = 1; j < count; ++j) {
                if (i == strs[j].size() || strs[j][i] != c) {
                    return strs[0].substr(0, i);
                }
            }
        }
        return strs[0];
    }
};
```
### 分治
$LCP$ 的计算满足结合律：
$$
LCP(S_1,...,S_n) = LCP(LCP(S_1,...,S_k), LCP(S_{k+1}, S_n)),   1 < k < n
$$
<img src="https://assets.leetcode-cn.com/solution-static/14/14_fig3.png" width="600" height="400" align="center">
``` c++
class Solution {
public:
    string longestCommonPrefix(vector<string>& strs) {
        if (!strs.size()) {
            return "";
        }
        else {
            return longestCommonPrefix(strs, 0, strs.size() - 1);
        }
    }

    string longestCommonPrefix(const vector<string>& strs, int start, int end) {
        if (start == end) {
            return strs[start];
        }
        else {
            int mid = (start + end) / 2;
            string lcpLeft = longestCommonPrefix(strs, start, mid);
            string lcpRight = longestCommonPrefix(strs, mid + 1, end);
            return commonPrefix(lcpLeft, lcpRight);
        }
    }

    string commonPrefix(const string& lcpLeft, const string& lcpRight) {
        int minLength = min(lcpLeft.size(), lcpRight.size());
        for (int i = 0; i < minLength; ++i) {
            if (lcpLeft[i] != lcpRight[i]) {
                return lcpLeft.substr(0, i);
            }
        }
        return lcpLeft.substr(0, minLength);
    }
};
```
### 二分查找
显然，最长公共前缀的长度不会超过字符串数组中的最短字符串的长度。用 $minLength$ 表示字符串数组中的最短字符串的长度，则可以在 $[0,minLength]$ 的范围内通过二分查找得到最长公共前缀的长度。每次取查找范围的中间值 $mid$，判断每个字符串的长度为 $mid$ 的前缀是否相同，如果相同则最长公共前缀的长度一定大于或等于 $mid$，如果不相同则最长公共前缀的长度一定小于 $mid$，通过上述方式将查找范围缩小一半，直到得到最长公共前缀的长度。示意图如下：
<img src="https://assets.leetcode-cn.com/solution-static/14/14_fig4.png" width="600" height="400" align="center">
``` c++
class Solution {
public:
    string longestCommonPrefix(vector<string>& strs) {
        if (!strs.size()) {
            return "";
        }
        int minLength = min_element(strs.begin(), strs.end(), [](const string& s, const string& t) {return s.size() < t.size();})->size();
        int low = 0, high = minLength;
        while (low < high) {
            int mid = (high - low + 1) / 2 + low;
            if (isCommonPrefix(strs, mid)) {
                low = mid;
            }
            else {
                high = mid - 1;
            }
        }
        return strs[0].substr(0, low);
    }

    bool isCommonPrefix(const vector<string>& strs, int length) {
        string str0 = strs[0].substr(0, length);
        int count = strs.size();
        for (int i = 1; i < count; ++i) {
            string str = strs[i];
            for (int j = 0; j < length; ++j) {
                if (str0[j] != str[j]) {
                    return false;
                }
            }
        }
        return true;
    }
};
```
## 三数之和
### 排序+双指针
题目中要求找到所有**不重复**且和为 0 的三元组，这个**不重复**的要求使得我们无法简单地使用三重循环枚举所有的三元组。**不重复**的本质是什么？我们保持三重循环的大框架不变，只需要保证：
- 第二重循环枚举到的元素不小于当前第一重循环枚举到的元素；
- 第三重循环枚举到的元素不小于当前第二重循环枚举到的元素。
也就是说，我们枚举的三元组 $(a, b, c)$ 满足$a \leqslant b \leqslant c$, 从而减少了重复。要实现这一点，我们可以将数组中的元素从小到大进行排序，随后使用普通的三重循环就可以满足上面的要求。同时，对于每一重循环而言，相邻两次枚举的元素不能相同，否则也会造成重复。此时算法的复杂度依然为 $O(N^3)$, 可以继续优化。
如果我们固定了前两重循环枚举到的元素 a 和 b，那么只有唯一的 c 满足 $a+b+c=0$。当第二重循环往后枚举一个元素 $b'$时，由于 $b' > b$，那么满足 $a+b'+c'=0$ 的 $c'$ 一定有 $c' < c$ 即 $c'$ 在数组中一定出现在 c 的左侧。也就是说，我们可以**从小到大枚举 b**，同时**从大到小枚举 c**，即`第二重循环和第三重循环实际上是并列的关系`。
我们就可以保持第二重循环不变，而将**第三重循环变成一个从数组最右端开始向左移动的指针**
``` c++
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        int n = nums.size();
        sort(nums.begin(), nums.end());
        vector<vector<int>> ans;
        // 枚举 a
        for (int first = 0; first < n; ++first) {
            // 需要和上一次枚举的数不相同
            if (first > 0 && nums[first] == nums[first - 1]) {
                continue;
            }
            // c 对应的指针初始指向数组的最右端
            int third = n - 1;
            int target = -nums[first];
            // 枚举 b
            for (int second = first + 1; second < n; ++second) {
                // 需要和上一次枚举的数不相同
                if (second > first + 1 && nums[second] == nums[second - 1]) {
                    continue;
                }
                // 需要保证 b 的指针在 c 的指针的左侧
                while (second < third && nums[second] + nums[third] > target) {
                    --third;
                }
                // 如果指针重合，随着 b 后续的增加
                // 就不会有满足 a+b+c=0 并且 b<c 的 c 了，可以退出循环
                if (second == third) {
                    break;
                }
                if (nums[second] + nums[third] == target) {
                    ans.push_back({nums[first], nums[second], nums[third]});
                }
            }
        }
        return ans;
    }
};
```
### 总结
当我们需要枚举数组中的两个元素时，如果我们发现随着第一个元素的递增，第二个元素是递减的，那么就可以使用**双指针**的方法，将枚举的时间复杂度从 $O(N^2)$ 减少至 $O(N)$。
## 电话号码的字母组合
### 回溯
回溯过程中维护一个字符串，表示**已有的字母排列**（如果未遍历完电话号码的所有数字，则已有的字母排列是不完整的）。该字符串初始为空。每次取电话号码的一位数字，从哈希表中获得该数字对应的所有可能的字母，并将其中的一个字母插入到已有的字母排列后面，然后继续处理电话号码的后一位数字，直到处理完电话号码中的所有数字，即得到一个完整的字母排列。然后进行回退操作，遍历其余的字母排列。
``` c++
class Solution {
public:
    vector<string> letterCombinations(string digits) {
        vector<string> combinations;
        if (digits.empty()) {
            return combinations;
        }
        unordered_map<char, string> phoneMap{
            {'2', "abc"},
            {'3', "def"},
            {'4', "ghi"},
            {'5', "jkl"},
            {'6', "mno"},
            {'7', "pqrs"},
            {'8', "tuv"},
            {'9', "wxyz"}
        };
        string combination;
        backtrack(combinations, phoneMap, digits, 0, combination);
        return combinations;
    }

    void backtrack(vector<string>& combinations, const unordered_map<char, string>& phoneMap, const string& digits, int index, string& combination) {
        if (index == digits.length()) {
            combinations.push_back(combination);
        } else {
            char digit = digits[index];
            const string& letters = phoneMap.at(digit);
            for (const char& letter: letters) {
                combination.push_back(letter);
                backtrack(combinations, phoneMap, digits, index + 1, combination);
                combination.pop_back();
            }
        }
    }
};
```
#### 复杂度分析
- 时间复杂度：$O(3^m \times 4^n)$；$m$ 是输入中对应 $3$ 个字母的数字个数（包括数字 $2、3、4、5、6、8$），$n$ 是输入中对应 $4$ 个字母的数字个数（包括数字 $7、9$），$m+n$ 是输入数字的总个数。
- 空间复杂度：$O(m+n)$；与递归层数有关
## 删除链表的倒数第N个结点
在对链表进行操作时，一种常用的技巧是添加一个**哑节点(dummy node)**，它的 $next$ 指针指向链表的头节点。这样一来，我们就不需要对头节点进行特殊的判断了。
``` c++
ListNode *dummy = new ListNode(0, head);
```
### 朴素想法——遍历计算链表长度
首先从头节点开始对链表进行一次遍历，得到链表的长度 $L$。随后我们再从**头节点(即哑节点)**开始对链表进行一次遍历，当遍历到第 $L-n+1$ 个节点时，它就是我们需要删除的节点。
``` c++
class Solution {
public:
    int getLength(ListNode* head) {
        int length = 0;
        while (head) {
            ++length;
            head = head->next;
        }
        return length;
    }

    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode* dummy = new ListNode(0, head);
        int length = getLength(head);
        ListNode* cur = dummy;
        for (int i = 1; i < length - n + 1; ++i) {
            cur = cur->next;
        }
        cur->next = cur->next->next;
        ListNode* ans = dummy->next;
        delete dummy;
        return ans;
    }
};
```
### 栈
我们也可以在遍历链表的同时将所有节点依次入栈。根据栈**先进后出**的原则，我们弹出栈的第 $n$ 个节点就是需要删除的节点，并且目前栈顶的节点就是待删除节点的前驱节点。这样一来，删除操作就变得十分方便了。
``` c++
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode* dummy = new ListNode(0, head);
        stack<ListNode*> stk;
        ListNode* cur = dummy;
        while (cur) {
            stk.push(cur);
            cur = cur->next;
        }
        for (int i = 0; i < n; ++i) {
            stk.pop();
        }
        ListNode* prev = stk.top();
        prev->next = prev->next->next;
        ListNode* ans = dummy->next;
        delete dummy;
        return ans;
    }
};
```
### 双指针
我们可以使用两个指针 *first* 和 *second* 同时对链表进行遍历，并且 *first* 比 *second* 超前 $n$ 个节点。当 *firs*t 遍历到链表的末尾时，*second* 就恰好处于倒数第 $n$ 个节点。
如果我们能够得到的是倒数第 $n$ 个节点的前驱节点而不是倒数第 $n$ 个节点的话，删除操作会更加方便。因此我们可以考虑在初始时将 *second* 指向哑节点，其余的操作步骤不变。这样一来，当 *first* 遍历到链表的末尾时，*second* 的下一个节点就是我们需要删除的节点。示意图如下：
<img src="https://assets.leetcode-cn.com/solution-static/19/p3.png" width="600" height="400" align="center">
``` c++
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode* dummy = new ListNode(0, head);
        ListNode* first = head;
        ListNode* second = dummy;
        for (int i = 0; i < n; ++i) {
            first = first->next;
        }
        while (first) {
            first = first->next;
            second = second->next;
        }
        second->next = second->next->next;
        ListNode* ans = dummy->next;
        delete dummy;  // 有new就应该有delete 
        return ans;
    }
};
```
## 合并两个有序链表
### 递归
#### 思想
可以如下递归地定义两个链表里的 *merge* 操作：
$$
\begin{cases} list1[0] + merge(list[1:], list2) & list1[0]<list2[0] \\ list2[0] + merge(list1, list2[1:]) & otherwise \end{cases}
$$
如果 $l1$ 或者 $l2$ 一开始就是空链表 ，那么没有任何操作需要合并，所以我们只需要返回非空链表。否则，我们要判断 $l1$ 和 $l2$ 哪一个链表的头节点的值更小，然后递归地决定下一个添加到结果里的节点。如果两个链表有一个为空，递归结束。
``` c++
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        if (l1 == nullptr) {
            return l2;
        } else if (l2 == nullptr) {
            return l1;
        } else if (l1->val < l2->val) {
            l1->next = mergeTwoLists(l1->next, l2);
            return l1;
        } else {
            l2->next = mergeTwoLists(l1, l2->next);
            return l2;
        }
    }
};
```
#### 复杂度分析
- 时间复杂度：$O(n+m)$
- 空间复杂度：$O(n+m)$
### 迭代
设定一个哨兵节点 $prehead$，这可以在最后让我们比较容易地返回合并后的链表。我们维护一个 $prev$ 指针，我们需要做的是调整它的 $next$ 指针。然后，我们重复以下过程，直到 $l1$ 或者 $l2$ 指向了 $null$：如果 $l1$ 当前节点的值小于等于 $l2$ ，我们就把 $l1$ 当前的节点接在 $prev$ 节点的后面同时将 $l1$ 指针往后移一位。否则，我们对 $l2$ 做同样的操作。不管我们将哪一个元素接在了后面，我们都需要把 $prev$ 向后移一位。
``` c++
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        ListNode* preHead = new ListNode(-1);

        ListNode* prev = preHead;
        while (l1 != nullptr && l2 != nullptr) {
            if (l1->val < l2->val) {
                prev->next = l1;
                l1 = l1->next;
            } else {
                prev->next = l2;
                l2 = l2->next;
            }
            prev = prev->next;
        }

        // 合并后 l1 和 l2 最多只有一个还未被合并完，我们直接将链表末尾指向未合并完的链表即可
        prev->next = l1 == nullptr ? l2 : l1;

        return preHead->next;
    }
};
```
## 括号生成
### 回溯法
只在序列仍然保持有效时才添加 $($ or $)$，可以通过跟踪到目前为止放置的左括号和右括号的数目来做到这一点，如果左括号数量不大于 $n$，我们可以放一个左括号。如果右括号数量小于左括号的数量，我们可以放一个右括号。
``` c++
class Solution {
    void backtrack(vector<string>& ans, string& cur, int open, int close, int n) {
        if (cur.size() == n * 2) {
            ans.push_back(cur);
            return;
        }
        if (open < n) {
            cur.push_back('(');
            backtrack(ans, cur, open + 1, close, n);
            cur.pop_back();
        }
        if (close < open) {
            cur.push_back(')');
            backtrack(ans, cur, open, close + 1, n);
            cur.pop_back();
        }
    }
public:
    vector<string> generateParenthesis(int n) {
        vector<string> result;
        string current;
        backtrack(result, current, 0, 0, n);
        return result;
    }
};
```
## 合并K个升序列表
### 分治法
<img src="https://pic.leetcode-cn.com/6f70a6649d2192cf32af68500915d84b476aa34ec899f98766c038fc9cc54662-image.png" width = "600" height="400" align="center">
``` c++
class Solution {
ListNode* mergeKLists(const vector<ListNode*> &lists, int start, int end) {
    if (start == end) {
        return lists[start];
    }else
    {
        int mid = (start + end) / 2;
        ListNode *mergeLeft = mergeKLists(lists, start, mid);
        ListNode *mergeRight = mergeKLists(lists, mid+1, end);
        return mergeTwoLists(mergeLeft, mergeRight);
    }
    
}

ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
    ListNode *dummy = new ListNode(0);
    ListNode *tmp = dummy;
    while (l1 && l2) {
        if (l1->val <= l2->val) {
            tmp->next = l1;
            l1 = l1->next;
        }else
        {
            tmp->next = l2;
            l2 = l2->next;
        }
        tmp = tmp->next;
    }

    if (l1 == nullptr)
        tmp->next = l2;
    else
        tmp->next = l1;
    
    ListNode *ans = dummy->next;
    delete dummy;
    return ans;
}
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
    int k = lists.size();
    if (k == 0)
        return nullptr;
    else
        return mergeKLists(lists, 0, k-1);    
}
};
```
- 时间复杂度：第一轮合并 $\frac{k}{2}$ 组链表，每一组的时间代价是 $O(2n)$；第二轮合并 $\frac{k}{4}$ 组链表，每一组的时间代价是 $O(4n)$...渐进时间复杂度是 $O(kn \times \log_2 k)$
- 空间复杂度：递归会调用 $O(\log_2 k)$ 的栈空间
### 优先队列
维护当前每个链表没有被合并的元素的最前面一个，$k$ 个链表就最多有 $k$ 个满足这样条件的元素，每次在这些元素里面选取 $val$ 属性最小的元素合并到答案中。在选取最小元素的时候，我们可以用`优先队列`来优化这个过程。
``` c++
class Solution {
public:
    struct Status {
        int val;
        ListNode *ptr;
        bool operator < (const Status &rhs) const {
            return val > rhs.val;
        }
    };

    priority_queue <Status> q;

    ListNode* mergeKLists(vector<ListNode*>& lists) {
        for (auto node: lists) {
            if (node) q.push({node->val, node});
        }
        ListNode head, *tail = &head;
        while (!q.empty()) {
            auto f = q.top();
            q.pop();
            tail->next = f.ptr; 
            tail = tail->next;
            if (f.ptr->next) q.push({f.ptr->next->val, f.ptr->next});
        }
        return head.next;
    }
};
```
- 时间复杂度：考虑优先队列中的元素不超过 $k$ 个，那么插入和删除的时间代价为 $O(\log_2 k)$，这里最多有 $kn$ 个点，对于每个点都被插入删除各一次，故总的时间代价即渐进时间复杂度为 $O(kn \times \log _2 k)$。
- 空间复杂度：优先队列中的元素不超过 $k$ 个，故渐进空间复杂度为 $O(k)$。
## 两两交换链表中的节点
### 迭代
<img src="https://pic.leetcode-cn.com/42c91b69e3f38d63a0d0153c440724e69bd2d24b95091b4dcc5c68172f8f4e1e-%E8%BF%AD%E4%BB%A3.gif" width="600" height="400" align="center">
``` c++
class Solution {
    public ListNode swapPairs(ListNode head) {
        ListNode dummyHead = new ListNode(0);
        dummyHead.next = head;
        ListNode temp = dummyHead;
        while (temp.next != null && temp.next.next != null) {
            ListNode node1 = temp.next;
            ListNode node2 = temp.next.next;
            temp.next = node2;
            node1.next = node2.next;
            node2.next = node1;
            temp = node1;
        }
        return dummyHead.next;
    }
}
```
### 递归
- 递归的终止条件是链表中没有节点，或者链表中只有一个节点，此时无法进行交换。
- 如果链表中至少有两个节点，则在两两交换链表中的节点之后，原始链表的头节点变成新的链表的第二个节点`指向下一层递归函数`，原始链表的第二个节点变成新的链表的头节点。链表中的其余节点的两两交换可以递归地实现。在对链表中的其余节点递归地两两交换之后，更新节点之间的指针关系，即可完成整个链表的两两交换。
<img src="https://pic.leetcode-cn.com/7ae491344608971d449add1e069aa143ee264b07a9bb8a1950e08dcf8d8a1ff9-%E9%80%92%E5%BD%92.gif" width="600" height="400" align="center">
``` c++
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        if (head == nullptr || head->next == nullptr) {
            return head;
        }
        ListNode* newHead = head->next;
        head->next = swapPairs(newHead->next);
        newHead->next = head;
        return newHead;
    }
};
```
## 反转链表
### 迭代
- 在遍历链表时，将当前节点的 $next$ 指针改为指向前一个节点。由于节点没有引用其前一个节点，因此必须事先存储其前一个节点。在更改引用之前，还需要存储后一个节点。最后返回新的头节点。
<img src="https://pic.leetcode-cn.com/7d8712af4fbb870537607b1dd95d66c248eb178db4319919c32d9304ee85b602-%E8%BF%AD%E4%BB%A3.gif" width="600" height="400" align="center" >
动画演示中其实省略了一个 $tmp$变量，这个 $tmp$ 变量会将 $cur$ 的下一个节点保存起来
``` c++
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* prev = nullptr;
        ListNode* curr = head;
        while (curr) {
            ListNode* next = curr->next;
            curr->next = prev;
            prev = curr;
            curr = next;
        }
        return prev;
    }
};
```
### 递归
<img src="https://pic.leetcode-cn.com/dacd1bf55dec5c8b38d0904f26e472e2024fc8bee4ea46e3aa676f340ba1eb9d-%E9%80%92%E5%BD%92.gif" wifth="600" height="400" align="center">
``` c++
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        if (!head || !head->next) {
            return head;
        }
        ListNode* newHead = reverseList(head->next);
        head->next->next = head;
        head->next = nullptr;
        return newHead;
    }
};
```
## 整数除法
### 倍增乘法
<img src="https://pic.leetcode-cn.com/1602148365-shUMvq-image.png" width="800" height="500">

``` c++
class Solution {
public:
    int divide(int dividend, int divisor) {
        if(dividend == 0) return 0;
        if(divisor == 1) return dividend;
        if(divisor == -1){
            if(dividend>INT_MIN) 
                return -dividend;// 只要不是最小的那个整数，都是直接返回相反数就好啦
            else 
                return INT_MAX;// 是最小的那个，那就返回最大的整数啦
        }
        int sign = 1; 
        if((dividend>0&&divisor<0) || (dividend<0&&divisor>0)){
            sign = -1;
        }
        dividend=-abs(dividend);//都转换为负数，避免溢出
        divisor=-abs(divisor);
        int res = div(dividend,divisor);
        if(sign>0)
            return res;
        else
            return -res;
    }
    int div(long a, long b){  // 似乎精髓和难点就在于下面这几句
        if(a>b) return 0;
        int count = 1;
        int tb = b; // 在后面的代码中不更新b
        while((tb-a+tb)>=0){
            count = count + count; // 最小解翻倍
            tb = tb+tb; // 当前测试的值也翻倍
        }
        return count + div(a-tb,b);
    }
};
```
## 串联所有单词的子串
### 滑动窗口+HashMap
首先，最直接的思路，判断每个子串是否符合，符合就把下标保存起来，最后返回即可。
<img src="https://pic.leetcode-cn.com/d79049a498051b0ee12ac60d41810c335c6483ecc2feb485b414ddd7a21af121-image.png" width="600" height="400" align="center">
由于子串包含的单词顺序并不需要固定，当单词数量增加，时间复杂度呈指数增加。考虑用`两个HashMap`来解决。首先，我们把所有的单词存到 **HashMap** 里，**key** 直接存单词，**value** 存单词出现的个数。然后扫描子串的单词，如果当前扫描的单词在之前的 **HashMap** 中，就把该单词存到新的 **HashMap** 中，并判断新的 **HashMap** 中该单词的 **value** 是不是大于之前的 **HashMap** 该单词的 **value** ，如果大了，就代表该子串不是我们要找的，接着判断下一个子串就可以了。如果不大于，那么我们接着判断下一个单词的情况。子串扫描结束，如果子串的全部单词都符合，那么该子串就是我们找的其中一个。具体例子如下：
1. 把 $words$ 中的单词存到一个 **HashMap**中：
<img src="https://pic.leetcode-cn.com/b8cb6bc68dc678d742b758afd94ffb63bb84818464d5c8df9b03b7a1067c4537-image.png" width="800" height="360" style="zoom:80%;" >
2. 然后遍历子串的每个单词：
<img src="https://pic.leetcode-cn.com/b0837f8651d20e328fd81b9137a77a1fe42b4f17a1bc76341b781cff0f92496e-image.png" >
<img src="https://pic.leetcode-cn.com/f9710845a8a8c2202cb1bedd8be9415fa81bf5f1169f7ef990e91c45fec7cf02-image.png" >
<img src="https://pic.leetcode-cn.com/6839efd1535699566f5803e56b9476c319ab175dddb948ed74007fade746a9e8-image.png" >
3. 比较此时 $foo$ 的 **value** 和 **HashMap1** 中 $foo$ 的 **value**，$3 > 2$，所以表明该字符串不符合，然后判断下个子串就好了。如果子串中的单词不在 **HashMap** 中也表明当前子串不符合。
``` c++
class Solution {
public:
    vector<int> findSubstring(string s, vector<string>& words) {
        vector<int> ret;
        if(words.size() == 0)//判断words为空,因为下面用到了words[0]
            return ret;
        
        int word_size = words[0].size();
        int word_num = words.size();
        
        unordered_map<string,int> m1;//构造hashmap
        for(int i=0;i<word_num;i++)
            m1[words[i]]++;
        
        unordered_map<string,int> m2;
        for(int i=0; (i + word_size * word_num) <= s.size() ; i++){//截取的s符合题意
            int j = 0;
            for(j=i;j < (i + word_size * word_num) ; j=j+word_size){//分段判断
                string temp_str = s.substr(j,word_size);
                if(m1[temp_str] == 0){//m1中没有截取的串，直接跳出
                    break;
                }else{
                    m2[temp_str]++;
                    if(m1[temp_str] < m2[temp_str])//重复次数过多，也跳出
                        break;
                }
            }
            
            if(j == (i + word_size * word_num))//每一段都符合，则加入答案
                ret.push_back(i);
            
            m2.clear();//清空m2
        }
        return ret;
    }
};
```
- 时间复杂度：假设 $s$ 的长度为 $n$，**words** 中有 $m$ 个单词，那么时间复杂度为 $O(m*n)$
### 解法二：滑动窗口优化
解法一中每次移动一个字符。我们可以优化为每次移动一个单词的长度，假设单词长度为3。这样所有的移动被分为了3类。
- 情况一：子串完全匹配
<img src="https://pic.leetcode-cn.com/b3c068a26c1c78c31283cbd3180eb9fc59a2a33676a5ec555813b032535dee6a-image.png">
在解法一中，对于 $i = 3$ 的子串，我们肯定是从第一个 **foo** 开始判断。但其实前两个 **foo** 都不用判断了 ，因为在判断上一个 $i = 0$ 的子串的时候我们已经判断过了。所以解法一中的 **HashMap2** 每次并不需要清空从 0 开始，而是可以只移除之前 $i = 0$ 子串的第一个单词 **bar** 即可，然后直接从箭头所指的 **foo** 开始就可以了。
- 情况二：判断过程中，出现不符合的单词
<img src="https://pic.leetcode-cn.com/5c2e75a2efbceffa66af036fe452c5addbc2d464149cd396897de22b0034fca5-image.png">
判断 $i = 0$ 的子串的时候，出现了 **the** ，并不在所给的单词中。所以此时 $i = 3$，$i = 6$ 的子串，我们其实并不需要判断了。我们直接判断 $i = 9$ 的情况就可以了。
- 情况三：符合单词的个数超了
<img src="https://pic.leetcode-cn.com/199f57d196ac73e3c922d94e12346fb88cca3ec0acc29eaf3a357ec2bdf5a29f-image.png">
此时我们只需要往后移动窗口，$i = 3$ 的子串将 **foo** 移除，此时子串中一定还是有两个 **bar**，所以该子串也一定不符合。接着往后移动，当之前的 **bar** 被移除后，此时 $i = 6$ 的子串，就可以接着按正常的方法判断了。
``` c++
class solution {
public:
        vector<int> findSubstring(string s, vector<string>& words) {
            if(words.empty()) return {};
            unordered_map<string,int> wordmap,smap;
            for(string word:words) wordmap[word]++;
            int wordlen = words[0].size();
            int wordnum = words.size();
            vector<int> ans;
            for(int k=0;k<wordlen;k++){
                int i=k,j=k;
                while(i<s.size()-wordnum*wordlen+1){
                    while(j<i+wordnum*wordlen){
                        string temp = s.substr(j,wordlen);
                        smap[temp]++;
                        j+=wordlen;
                        if(wordmap[temp]==0){//情况二，有words中不存在的单词
                            i=j;//对i加速
                            smap.clear();
                            break;
                        }
                        else if(smap[temp]>wordmap[temp]){//情况三，子串中temp数量超了
                            while(smap[temp]>wordmap[temp]){
                                smap[s.substr(i,wordlen)]--;
                                i+=wordlen;//对i加速
                            }
                            break;
                        }                   
                    }
                    //正确匹配，由于情况二和三都对i加速了，不可能满足此条件
                    if(j==i+wordlen*wordnum){
                        ans.push_back(i);
                        smap[s.substr(i,wordlen)]--;
                        i+=wordlen;//i正常前进
                    }
                }
                smap.clear();
            }
            return ans;
        }
};
```
### 总结
这道题自己写得时候一直想用**回溯**，这是没有完全理解题意，题目要求是找单词组合在字符串中出现的位置，而不是找合法的单词组合，所以**回溯**实际上增加了时间复杂性。
## 下一个排列——两遍扫描
### 思路
- 将一个左边的**较小数**与一个右边的**较大数**交换，以能够让当前排列变大，从而得到下一个排列。
- 同时我们要让这个**较小数**尽量靠右，而**较大数**尽可能小。当交换完成后，**较大数**右边的数需要按照升序重新排列。这样可以在保证新排列大于原来排列的情况下，使变大的幅度尽可能小。
### 解法
- 首先从后向前查找第一个顺序对 $(i,i+1)$，满足 $a[i] < a[i+1]$。这样**较小数**即为 $a[i]$。此时 $[i+1,n)$ 必然是下降序列。
- 如果找到了顺序对，那么在区间 $[i+1,n)$ 中从后向前查找第一个元素 $j$ 满足 $a[i] < a[j]$ 。这样**较大数**即为 $a[j]$。
- 交换 $a[i]$ 与 $a[j]$，此时可以证明区间 $[i+1,n)$ 必为降序。我们可以直接使用双指针反转区间 $[i+1,n)$ 使其变为升序，而无需对该区间进行排序。
<img src="https://assets.leetcode-cn.com/solution-static/31/31.gif">
``` c++
class Solution {
public:
    void nextPermutation(vector<int>& nums) {
        int i = nums.size() - 2;
        while (i >= 0 && nums[i] >= nums[i + 1]) {
            --i;
        }
        if (i >= 0) {
            int j = nums.size() - 1;
            while (j >= 0 && nums[i] >= nums[j]) {
                --j;
            }
            swap(nums[i], nums[j]);
        }
        reverse(nums.begin() + i + 1, nums.end());
    }
};
```
## 最长有效括号
### 动态规划
定义 $dp[i]$ 表示以下标 $i$ 字符结尾的最长有效括号的长度。我们将 $dp$ 数组全部初始化为 $0$ 。显然有效的子串一定以 $)$ 结尾，因此我们可以知道以 $($ 结尾的子串对应的 $dp$ 值必定为 $0$ ，我们只需要求解 $)$ 在 $dp$ 数组中对应位置的值。从前往后遍历字符串求解 \textit{dp}dp 值，我们每两个字符检查一次：
1. $s[i]=)$ 且 $s[i - 1] = ($，也就是字符串形如 $“……()”$，我们可以推出：
$$
dp[i] = dp[i-2]+2
$$
2. $s[i]=)$ 且 $s[i - 1] = )$，也就是字符串形如 $“……))”$，我们可以推出：如果 $s[i-dp[i-1]-1]=($，那么
$$
dp[i] = dp[i-dp[i-1]-2]+2
$$

最后的答案即为 $dp$ 数组中的最大值。
<img src="https://media.giphy.com/media/fUMCFzU9LXxyVMMjEV/giphy.gif" width="700" height="400">

``` c++
class Solution {
public:
    int longestValidParentheses(string s) {
        int maxans = 0, n = s.length();
        vector<int> dp(n, 0);
        for (int i = 1; i < n; i++) {
            if (s[i] == ')') {
                if (s[i - 1] == '(') {
                    dp[i] = (i >= 2 ? dp[i - 2] : 0) + 2;
                } else if (i - dp[i - 1] > 0 && s[i - dp[i - 1] - 1] == '(') {
                    dp[i] = dp[i - 1] + ((i - dp[i - 1]) >= 2 ? dp[i - dp[i - 1] - 2] : 0) + 2;
                }
                maxans = max(maxans, dp[i]);
            }
        }
        return maxans;
    }
};
```
### 栈
始终保持栈底元素为当前已经遍历过的元素中**最后一个没有被匹配的右括号的下标**，这样的做法主要是考虑了边界条件的处理，栈里其他元素维护左括号的下标：
- 对于遇到的每个 $($ ，我们将它的下标放入栈中
- 对于遇到的每个 $)$，我们先弹出栈顶元素表示匹配了当前右括号：
    - 如果栈为空，说明当前的右括号为没有被匹配的右括号，我们将其下标放入栈中来更新我们之前提到的**最后一个没有被匹配的右括号的下标**
    - 如果栈不为空，当前右括号的下标减去栈顶元素即为**以该右括号为结尾的最长有效括号的长度**。

然后从前往后遍历字符串更新答案即可。需要注意的是，如果一开始栈为空，第一个字符为左括号的时候我们会将其放入栈中，这样就不满足提及的**最后一个没有被匹配的右括号的下标**，为了保持统一，我们在一开始的时候往栈中放入一个值为 $−1$ 的元素。
<img src="https://media.giphy.com/media/2b3MxNcUEo1kFoJobG/giphy.gif" width="800" height="600" align="center">
``` c++
class Solution {
public:
    int longestValidParentheses(string s) {
        int maxans = 0;
        stack<int> stk;
        stk.push(-1);
        for (int i = 0; i < s.length(); i++) {
            if (s[i] == '(') {
                stk.push(i);
            } else {
                stk.pop();
                if (stk.empty()) {
                    stk.push(i);
                } else {
                    maxans = max(maxans, i - stk.top());
                }
            }
        }
        return maxans;
    }
};
```
### 左右两遍扫描
利用两个计数器 $left$ 和 $right$。
- 首先，我们从左到右遍历字符串，对于遇到的每个 $($，我们增加 $left$ 计数器，对于遇到的每个 $)$，我们增加 $right$ 计数器。每当 $left$ 计数器与 $right$ 计数器相等时，我们计算当前有效字符串的长度，并且记录目前为止找到的最长子字符串。当 $right$ 计数器比 $left$ 计数器大时，我们将 $left$ 和 $right$ 计数器同时变回 $0$。
- 这样的做法贪心地考虑了以当前字符下标结尾的有效括号长度，每次当右括号数量多于左括号数量的时候之前的字符我们都扔掉不再考虑，重新从下一个字符开始计算，但这样会漏掉一种情况，就是遍历的时候左括号的数量始终大于右括号的数量，即 $(()$ ，这种时候最长有效括号是求不出来的。
- 解决的方法也很简单，我们只需要从右往左遍历用类似的方法计算即可，只是这个时候判断条件反了过来。
<img src="https://media.giphy.com/media/lfnjM5cMQVkpNJltr8/giphy.gif" width="800" height="500">
``` c++
class Solution {
public:
    int longestValidParentheses(string s) {
        int left = 0, right = 0, maxlength = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s[i] == '(') {
                left++;
            } else {
                right++;
            }
            if (left == right) {
                maxlength = max(maxlength, 2 * right);
            } else if (right > left) {
                left = right = 0;
            }
        }
        left = right = 0;
        for (int i = (int)s.length() - 1; i >= 0; i--) {
            if (s[i] == '(') {
                left++;
            } else {
                right++;
            }
            if (left == right) {
                maxlength = max(maxlength, 2 * left);
            } else if (left > right) {
                left = right = 0;
            }
        }
        return maxlength;
    }
};
```
## 搜索旋转排序数组——二分查找
### 思路
我们将数组从中间分开成左右两部分的时候，一定有一部分的数组是有序的。拿示例来看，我们从 $6$ 这个位置分开以后数组变成了 $[4, 5, 6]$ 和 $[7, 0, 1, 2]$ 两个部分，其中左边 $[4, 5, 6]$ 这个部分的数组是有序的，其他也是如此。
### 解法
这启示我们可以在常规二分查找的时候查看当前 $mid$ 为分割位置分割出来的两个部分 $[l, mid]$ 和 $[mid + 1, r]$ 哪个部分是有序的，并根据有序的那个部分确定我们该如何改变二分查找的上下界，因为我们能够根据有序的那部分判断出 $target$ 在不在这个部分：
    - 如果 $[l, mid - 1]$ 是有序数组，且 $target$ 的大小满足 $[nums[l],nums[mid])$，则我们应该将搜索范围缩小至 $[l, mid - 1]$，否则在 $[mid + 1, r]$ 中寻找。
    - 如果 $[mid, r]$ 是有序数组，且 $target$ 的大小满足 $(nums[mid+1],nums[r]]$，则我们应该将搜索范围缩小至 $[mid + 1, r]$，否则在 $[l, mid - 1]$ 中寻找。

<img src="https://assets.leetcode-cn.com/solution-static/33/33_fig1.png">
二分查找**迭代**写法如下：
``` c++
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int n = (int)nums.size();
        if (!n) {
            return -1;
        }
        if (n == 1) {
            return nums[0] == target ? 0 : -1;
        }
        int l = 0, r = n - 1;
        while (l <= r) {
            int mid = (l + r) / 2;
            if (nums[mid] == target) return mid;
            if (nums[0] <= nums[mid]) {
                if (nums[0] <= target && target < nums[mid]) {
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
        return -1;
    }
};
```
二分查找**递归**写法如下：
``` c++
class Solution {
public:

    int searchSub(vector<int>& nums, int left, int right, int target) {

        if (left == right) {
            return nums[left] == target? left : -1;
        }

        int mid = (left + right) / 2;
        if (nums[left] <= nums[mid]) {
            if (target >= nums[left] && target <= nums[mid])
                return searchSub(nums, left, mid, target);
            else
                return searchSub(nums, mid+1, right, target);
        }else
        {
            if (target >= nums[mid] && target <= nums[right])
                return searchSub(nums, mid, right, target);
            else
                return searchSub(nums, left, mid-1, target);
        }
    }

    int search(vector<int>& nums, int target) {
        int len = nums.size();

        int l = 0, r = len-1;
        int ans = searchSub(nums, l, r, target);

        return ans;
    }
};
```
## 在排序数组中查找元素的第一个和最后一个位置
### 思路
考虑 $target$ 开始和结束位置，其实我们要找的就是数组中**第一个等于 $target$ 的位置**（记为 $leftIdx$）和**第一个大于 $target$ 的位置减一**（记为 $rightIdx$）。
### 解法
二分查找中，寻找 $leftIdx$ 即为在数组中寻找第一个大于等于 $target$ 的下标，寻找 $rightIdx$ 即为在数组中寻找第一个大于 $target$ 的下标，然后将下标减一。两者的判断条件不同，为了代码的复用，我们定义 $binarySearch(nums, target, lower)$ 表示在 $nums$ 数组中二分查找 $target$ 的位置，如果 $lower$ 为 $true$，则查找第一个大于等于 $target$ 的下标，否则查找第一个大于 $target$ 的下标。
最后，因为 $target$ 可能不存在数组中，因此我们需要重新校验我们得到的两个下标 $leftIdx$ 和 $rightIdx$，看是否符合条件，如果符合条件就返回 $[leftIdx,rightIdx]$，不符合就返回 $[−1,−1]$。
``` c++
class Solution { 
public:
    int binarySearch(vector<int>& nums, int target, bool lower) {
        int left = 0, right = (int)nums.size() - 1, ans = (int)nums.size();
        while (left <= right) {
            int mid = (left + right) / 2;
            if (nums[mid] > target || (lower && nums[mid] >= target)) {
                right = mid - 1;
                ans = mid;
            } else {
                left = mid + 1;
            }
        }
        return ans;
    }

    vector<int> searchRange(vector<int>& nums, int target) {
        int leftIdx = binarySearch(nums, target, true);
        int rightIdx = binarySearch(nums, target, false) - 1;
        if (leftIdx <= rightIdx && rightIdx < nums.size() && nums[leftIdx] == target && nums[rightIdx] == target) {
            return vector<int>{leftIdx, rightIdx};
        } 
        return vector<int>{-1, -1};
    }
};
```
## 有效的数独
- 如何枚举子数独?
> 可以使用 $box_{index} = (row / 3) * 3 + columns / 3$

<img  src="https://pic.leetcode-cn.com/2b141392e2a1811d0e8dfdf6279b1352e59fad0b3961908c6ff9412b6a7e7ccf-image.png" width="400" height="400">
- 算法步骤：
    - 遍历数独
    - 检查看到每个单元格值是否已经在当前的行 / 列 / 子数独中出现过
    - 返回$true$

``` c++
class Solution {
public:
    bool isValidSudoku(vector<vector<char>>& board) {
        int row[9][9]={0};
        int col[9][9]={0};
        int grid[9][9]={0};
        int n;

        for (int i=0;i<9;i++){
            for (int j=0;j<9;j++){
                if(board[i][j]!='.'){
                    n = board[i][j]-'1';

                    if (row[i][n]!=0 || col[j][n]!=0 || grid[(i/3)*3+(j/3)][n]!=0)
                        return false;

                    row[i][n]=1;
                    col[j][n]=1;
                    grid[(i/3)*3+(j/3)][n]=1;
                }
            }
        }
        return true;
    }
};
```
## 组合总和——搜索回溯
对于这类寻找所有可行解的题，我们都可以尝试用**搜索回溯**的方法来解决。
### 思路
定义递归函数 $dfs(target, combine, idx)$ 表示当前在 $candidates$ 数组的第 $idx$ 位，还剩 $target$ 要组合，已经组合的列表为 $combine$。递归的终止条件为 $target <= 0$ 或者 $candidates$ 数组被全部用完。那么在当前的函数中，每次我们可以选择跳过不用第 $idx$ 个数，即执行 $dfs(target, combine, idx + 1)$。也可以选择使用第 $idx$ 个数，即执行 $dfs(target - candidates[idx], combine, idx)$，注意到每个数字可以被无限制重复选取，因此搜索的下标仍为 $idx$。
`自己写的代码`：
``` c++
class Solution {
public:

    void backtrack(set<vector<int>> &filter, vector<int> &combination, vector<int> &candidates, int i, int sum, int target) {
        sum += candidates[i];
        if (sum == target) {
            combination.push_back(candidates[i]);
            filter.insert(combination);
            combination.pop_back();
        }else if (sum < target) {
            combination.push_back(candidates[i]);
            for (int p = i; p < candidates.size(); ++p) {
                backtrack(filter, combination, candidates, p, sum, target);
            }            
            combination.pop_back();
        }else
        {
            return;
        }
        
    }

    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        sort(candidates.begin(), candidates.end());
        vector<vector<int>> ans;
        set<vector<int>> filter;
        vector<int> combination;
        for (int i = 0; i < candidates.size(); ++i) {
            backtrack(filter, combination, candidates, i, 0, target);
        }
            

        for(auto sum : filter)
            ans.push_back(sum);

        return ans;
    }
};
```
如果我们将整个搜索过程用一个树来表达，即如下图呈现，每次的搜索都会延伸出两个分叉，直到递归的终止条件，这样我们就能不重复且不遗漏地找到所有可行解：
<img src="https://assets.leetcode-cn.com/solution-static/39/39_fig1.png"  width="800" height="500">
``` c++
class Solution {
public:
    void dfs(vector<int>& candidates, int target, vector<vector<int>>& ans, vector<int>& combine, int idx) {
        if (idx == candidates.size()) {
            return;
        }
        if (target == 0) {
            ans.emplace_back(combine);
            return;
        }
        // 直接跳过
        dfs(candidates, target, ans, combine, idx + 1);
        // 选择当前数
        if (target - candidates[idx] >= 0) {
            combine.emplace_back(candidates[idx]);
            dfs(candidates, target - candidates[idx], ans, combine, idx);
            combine.pop_back();
        }
    }

    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<vector<int>> ans;
        vector<int> combine;
        dfs(candidates, target, ans, combine, 0);
        return ans;
    }
};
```
- 时间复杂度：$O(S)$
- 空间复杂度：$O(target)$
## 组合总和2
在求出组合的过程中就进行去重的操作。我们可以考虑将相同的数放在一起进行处理，也就是说，如果数 $x$ 出现了 $y$ 次，那么在递归时一次性地处理它们，即分别调用选择 $0, 1, \cdots, y$ 次 $x$ 的递归函数。这样我们就不会得到重复的组合。具体地：
- 我们使用一个哈希映射（HashMap）统计数组 $candidates$ 中每个数出现的次数。在统计完成之后，我们将结果放入一个列表 $freq$ 中，方便后续的递归使用。
    - 列表 $freq$ 的长度即为数组 $candidates$ 中不同数的个数。其中的每一项对应着哈希映射中的一个键值对，即某个数以及它出现的次数。
- 在递归时，对于当前的第 $pos$ 个数，它的值为$freq[pos][0]$，出现的次数为 $freq[pos][1]$，那么我们可以调用：
$$
dfs(pos+1,rest − i \times freq[pos][0])
$$
    即我们选择了这个数 $i$ 次。这里 $i$ 不能大于这个数出现的次数，并且 $i \times freq[pos][0]$ 也不能大于 $rest$。同时，我们需要将 $i$ 个 $freq[pos][0]$ 放入列表中。

这样一来，我们就可以不重复地枚举所有的组合了。
一种比较常用的优化方法是，我们将 $freq$ 根据数从小到大排序，这样我们在递归时会先选择小的数，再选择大的数。
``` c++
class Solution {
private:
    vector<pair<int, int>> freq;
    vector<vector<int>> ans;
    vector<int> sequence;

public:
    void dfs(int pos, int rest) {
        if (rest == 0) {
            ans.push_back(sequence);
            return;
        }
        if (pos == freq.size() || rest < freq[pos].first) {
            return;
        }

        dfs(pos + 1, rest);

        int most = min(rest / freq[pos].first, freq[pos].second);
        for (int i = 1; i <= most; ++i) {
            sequence.push_back(freq[pos].first);
            dfs(pos + 1, rest - i * freq[pos].first);
        }
        for (int i = 1; i <= most; ++i) {
            sequence.pop_back();
        }
    }

    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        sort(candidates.begin(), candidates.end());
        for (int num: candidates) {
            if (freq.empty() || num != freq.back().first) {
                freq.emplace_back(num, 1);
            } else {
                ++freq.back().second;
            }
        }
        dfs(0, target);
        return ans;
    }
};
```
## 缺失的第一个正数
题目要求解法满足时间复杂度$O(N)$，空间复杂度$O(1)$
### 哈希表
对于一个长度为 $N$ 的数组，其中没有出现的最小正整数只能在 $[1,N+1]$ 中。这是因为如果 $[1,N]$ 都出现了，那么答案是 $N+1$，否则答案是 $[1,N]$ 中没有出现的最小正整数。这样一来，我们将所有在 $[1,N]$ 范围内的数放入哈希表，也可以得到最终的答案。而给定的数组恰好长度为 $N$，这让我们有了一种将数组设计成哈希表的思路：
> 对数组进行遍历，对于遍历到的数 $x$，如果它在 $[1,N]$ 的范围内，那么就将数组中的第 $x−1$ 个位置（注意：数组下标从 $0$ 开始）打上**标记**。在遍历结束之后，如果所有的位置都被打上了标记，那么答案是 $N+1$，否则答案是最小的没有打上标记的位置加 $1$。

那么如何设计这个「标记」呢？由于数组中的数没有任何限制，因此这并不是一件容易的事情。但我们可以继续利用上面的提到的性质：由于我们只在意 $[1,N]$ 中的数，因此我们可以先对数组进行遍历，把不在 $[1,N]$ 范围内的数修改成任意一个大于 $N$ 的数（例如 $N+1$）。这样一来，数组中的所有数就都是正数了，因此我们就可以将**标记**表示为**负号**。算法的流程如下：
- 将数组中所有小于等于 $0$ 的数修改为 $N+1$;
- 遍历数组中的每一个数 $x$，它可能已经被打了标记，因此原本对应的数为 $\|x\|$，其中 $\|$ 为绝对值符号。如果 $\|x\| \in [1, N]$，那么我们给数组中的第 $\|x\| - 1$ 个位置的数添加一个负号。注意如果它已经有负号，不需要重复添加；
- 在遍历完成之后，如果数组中的每一个数都是负数，那么答案是 $N+1$，否则答案是第一个正数的位置加 $1$。
<img src="https://assets.leetcode-cn.com/solution-static/41/41_fig1.png" width="780" height="400">
``` c++
class Solution {
public:
    int firstMissingPositive(vector<int>& nums) {
        int n = nums.size();
        for (int& num: nums) {
            if (num <= 0) {
                num = n + 1;
            }
        }
        for (int i = 0; i < n; ++i) {
            int num = abs(nums[i]);
            if (num <= n) {
                nums[num - 1] = -abs(nums[num - 1]);
            }
        }
        for (int i = 0; i < n; ++i) {
            if (nums[i] > 0) {
                return i + 1;
            }
        }
        return n + 1;
    }
};
```
### 置换
可以使用置换的方法，将给定的数组恢复成下面的形式：
> 如果数组中包含 $x \in [1, N]$，那么恢复后，数组的第 $x−1$ 个元素为 $x$。

在恢复后，数组应当有 $[1, 2, \cdots, N]$ 的形式，但其中有若干个位置上的数是错误的，每一个错误的位置就代表了一个缺失的正数。
对数组进行一次遍历，对于遍历到的数 $x=nums[i]$，如果 $x \in [1, N]$，我们就知道 $x$ 应当出现在数组中的$x−1$ 的位置，因此交换$nums[i]$ 和 $nums[x−1]$，这样 $x$ 就出现在了正确的位置。在完成交换后，新的 $nums[i]$ 可能还在 $[1,N]$ 的范围内，我们需要继续进行交换操作，直到 $x \notin [1, N]$
如果 $nums[i]$ 恰好与 $nums[x−1]$ 相等，那么就会无限交换下去。此时我们有 $nums[i]=x=nums[x−1]$，说明 $x$ 已经出现在了正确的位置。因此我们可以跳出循环，开始遍历下一个数。
``` c++
class Solution {
public:
    int firstMissingPositive(vector<int>& nums) {
        int n = nums.size();
        for (int i = 0; i < n; ++i) {
            while (nums[i] > 0 && nums[i] <= n && nums[nums[i] - 1] != nums[i]) {
                swap(nums[nums[i] - 1], nums[i]);
            }
        }
        for (int i = 0; i < n; ++i) {
            if (nums[i] != i + 1) {
                return i + 1;
            }
        }
        return n + 1;
    }
};
```
## 接雨水
对于下标 $i$，下雨后水能到达的最大高度等于下标 $i$ 两边的最大高度的最小值，下标 $i$ 处能接的雨水量等于下标 $i$ 处的水能到达的最大高度减去 $height[i]$。
### 动态规划
创建两个长度为 $n$ 的数组 $leftMax$ 和 $rightMax$。对于 $0 \le i < n$，$leftMax[i]$ 表示下标 $i$ 及其左边的位置中，$height$ 的最大高度，$rightMax[i]$ 表示下标 $i$ 及其右边的位置中，$height$ 的最大高度。
显然，$leftMax[0]=height[0]$，$rightMax[n−1]=height[n−1]$。两个数组的其余元素的计算如下：
- 当 $1 \le i \le n-1$ 时，$leftMax[i]=max(leftMax[i−1],height[i])$
- 当 $0 \le i \le n-2$ 时，$rightMax[i]=max(rightMax[i+1],height[i])$

因此可以正向遍历数组 $height$ 得到数组 $leftMax$ 的每个元素值，反向遍历数组 $height$ 得到数组 $rightMax$ 的每个元素值。
在得到数组 $leftMax$ 和 $rightMax$ 的每个元素值之后，对于 $0 \le i<n$，下标 $i$ 处能接的雨水量等于$min(leftMax[i],rightMax[i])−height[i]$。遍历每个下标位置即可得到能接的雨水总量。
动态规划思想示意图如下：
<img src="https://assets.leetcode-cn.com/solution-static/42/1.png" width="600" height="400">
``` c++
class Solution {
    public int trap(int[] height) {
        int n = height.length;
        if (n == 0) {
            return 0;
        }

        int[] leftMax = new int[n];
        leftMax[0] = height[0];
        for (int i = 1; i < n; ++i) {
            leftMax[i] = Math.max(leftMax[i - 1], height[i]);
        }

        int[] rightMax = new int[n];
        rightMax[n - 1] = height[n - 1];
        for (int i = n - 2; i >= 0; --i) {
            rightMax[i] = Math.max(rightMax[i + 1], height[i]);
        }

        int ans = 0;
        for (int i = 0; i < n; ++i) {
            ans += Math.min(leftMax[i], rightMax[i]) - height[i];
        }
        return ans;
    }
};
```

### 单调栈
维护一个单调栈，单调栈存储的是下标，满足从栈底到栈顶的下标对应的数组 $height$ 中的元素递减。
从左到右遍历数组，遍历到下标 $i$ 时，如果栈内至少有两个元素，记栈顶元素为 $top$，$top$ 的下面一个元素是 $left$，则一定有 $height[left]≥height[top]$。如果 $height[i]>height[top]$，则得到一个可以接雨水的区域，该区域的宽度是 $i−left−1$，高度是 $min(height[left],height[i])−height[top]$，根据宽度和高度即可计算得到该区域能接的雨水量。
为了得到 $left$，需要将 $top$ 出栈。在对 $top$ 计算能接的雨水量之后，$left$ 变成新的 $top$，重复上述操作，直到栈变为空，或者栈顶下标对应的 $height$ 中的元素大于或等于 $height[i]$。
在对下标 $i$ 处计算能接的雨水量之后，将 $i$ 入栈，继续遍历后面的下标，计算能接的雨水量。遍历结束之后即可得到能接的雨水总量。
<img src="https://media.giphy.com/media/mCwPEs7culiAICmPli/giphy.gif" width="600" height="400">
``` c++
class Solution {
public:
    int trap(vector<int>& height) {
        int ans = 0;
        stack<int> stk;
        int n = height.size();
        for (int i = 0; i < n; ++i) {
            while (!stk.empty() && height[i] > height[stk.top()]) {
                int top = stk.top();
                stk.pop();
                if (stk.empty()) {
                    break;
                }
                int left = stk.top();
                int currWidth = i - left - 1;
                int currHeight = min(height[left], height[i]) - height[top];
                ans += currWidth * currHeight;
            }
            stk.push(i);
        }
        return ans;
    }
};
```
### 双指针
动态规划的做法中，需要维护两个数组 $leftMax$ 和 $rightMax$，因此空间复杂度是 $O(n)$。
注意到下标 $i$ 处能接的雨水量由 $leftMax[i]$ 和 $rightMax[i]$ 中的最小值决定。由于数组 $leftMax$ 是从左往右计算，数组 $rightMax$ 是从右往左计算，因此可以使用双指针和两个变量代替两个数组。 
维护两个指针 $left$ 和 $right$，以及两个变量 $leftMax$ 和 $rightMax$，初始时 $left=0,right=n−1,leftMax=0,rightMax=0$。指针 $left$ 只会向右移动，指针 $right$ 只会向左移动，在移动指针的过程中维护两个变量 $leftMax$ 和 $rightMax$ 的值。
当两个指针没有相遇时，进行如下操作：
- 使用 $height[left]$ 和 $height[right]$ 的值更新 $leftMax$ 和 $rightMax$ 的值；
- 如果 $height[left]<height[right]$，则必有 $leftMax<rightMax$，下标 $left 处能接的雨水量等于 $leftMax−height[left]$，将下标 $left$ 处能接的雨水量加到能接的雨水总量，然后将 $left$ 加 $1$（即向右移动一位）；
- 如果 $height[left]≥height[right]$，则必有 $leftMax \ge rightMax$，下标 $right$ 处能接的雨水量等于 $rightMax−height[right]$，将下标 $right$ 处能接的雨水量加到能接的雨水总量，然后将 $right$ 减 $1$（即向左移动一位）。
当两个指针相遇时，即可得到能接的雨水总量。
<img src="https://media.giphy.com/media/eD7yeFW6dqbTqAsLIz/giphy.gif" width="600" height="400">
``` c++
class Solution {
public:
    int trap(vector<int>& height) {
        int ans = 0;
        int left = 0, right = height.size() - 1;
        int leftMax = 0, rightMax = 0;
        while (left < right) {
            leftMax = max(leftMax, height[left]);
            rightMax = max(rightMax, height[right]);
            if (height[left] < height[right]) {
                ans += leftMax - height[left];
                ++left;
            } else {
                ans += rightMax - height[right];
                --right;
            }
        }
        return ans;
    }
};
```
## 字符串乘法
### 模拟竖式乘法
从右往左遍历乘数，将乘数的每一位与被乘数相乘得到对应的结果，再将每次得到的结果累加。这道题中，被乘数是 $num1$，乘数是 $num2$。需要注意的是，$num2$ 除了最低位以外，其余的每一位的运算结果都需要补 $0$。
<img src="https://assets.leetcode-cn.com/solution-static/43/sol1.png" width="600" height="400">
``` c++
class Solution {
public:

    string stringAdd(string num1, string num2) {
        int i = num1.size()-1, j = num2.size()-1, carry = 0;
        string ans;
        while(i >= 0 || j >=0) {
            int x = i >= 0 ? num1.at(i)-'0' : 0;
            int y = j >= 0 ? num2.at(j)-'0' : 0;
            int sum = x + y + carry;
            ans.push_back(sum%10);
            carry = sum / 10;
            --i;
            --j;
        }
        if (carry > 0)
            ans.push_back(carry);
        reverse(ans.begin(), ans.end());
        for (auto &c : ans)
            c += '0';

        return ans;
    }

    string multiply(string num1, string num2) {
        if (num1 == "0" || num2 == "0")
            return "0";

        if (num2.size() > num1.size())
            swap(num1, num2);

        int m = num1.size(), n = num2.size();
        string ans;
        for (int i = n-1; i >= 0; --i) {
            string currMul;
            int carry = 0;
            for (int j = n-1; j > i; --j)
                currMul.push_back(0);
            int multiplier = num2.at(i) - '0';
            for (int k = m-1; k >= 0; --k) {
                int multiplicand = num1.at(k) - '0';
                int result = multiplicand * multiplier + carry;
                currMul.push_back(result%10);
                carry = result / 10;
            }
            while(carry != 0) {
                currMul.push_back(carry % 10);
                carry /= 10;
            }
            reverse(currMul.begin(), currMul.end());
            for (auto &c : currMul)
                c += '0';
            ans = stringAdd(ans, currMul);
        }
        return ans;
    }
};
```
## 通配符匹配
### 动态规划
#### 思路
用 $dp[i][j]$ 表示字符串 $s$ 的前 $i$ 个字符和模式 $p$ 的前 $j$ 个字符是否能匹配。在进行状态转移时，我们可以考虑模式 $p$ 的第 $j$ 个字符 $p_j$ 与之对应的是字符串 $s$ 中的第 $i$ 个字符 $s_i$：
- 如果 $p_j$ 是小写字母，那么 $s_i$ 必须也为相同的小写字母，状态转移方程为：
$$
dp[i][j] = (s_i与p_j相同) \wedge dp[i-1][j-1]
$$
- 如果 $p_j$ 是问号，那么对 $s_i$  没有任何要求，状态转移方程为：
$$
dp[i][j]=dp[i−1][j−1]
$$
- 如果 $p_j$  是星号，那么同样对 $s_i$  没有任何要求，但是星号可以匹配零或任意多个小写字母，因此状态转移方程分为两种情况，即使用或不使用这个星号：
$$
dp[i][j]=dp[i][j−1] \vee dp[i−1][j]
$$
最终的状态转移方程如下：
$$
dp[i][j] = \begin{cases} dp[i-1][j-1], & s_i与p_j相同或者p_j是问号 \\ dp[i][j-1] \vee dp[i-1][j], & p_j是星号 \\ False, & 其它情况 \end{cases}
$$
#### 细节
只有确定了边界条件，才能进行动态规划。所有的 $dp[0][j]$ 和 $dp[i][0]$ 都是边界条件，因为它们涉及到空字符串或者空模式的情况，这是我们在状态转移方程中没有考虑到的：
- $dp[0][0]=True$，即当字符串 $s$ 和模式 $p$ 均为空时，匹配成功；
- $dp[i][0]=False$，即空模式无法匹配非空字符串；
- $dp[0][j]$ 需要分情况讨论：因为星号才能匹配空字符串，所以只有当模式 $p$ 的前 $j$ 个字符均为星号时，$dp[0][j]$ 才为真。
我们可以发现，$dp[i][0]$ 的值恒为假，$dp[0][j]$ 在 $j$ 大于模式 $p$ 的开头出现的星号字符个数之后，值也恒为假，而 $dp[i][j]$ 的默认值（其它情况）也为假，因此在对动态规划的数组初始化时，我们就可以将所有的状态初始化为 $False$，减少状态转移的代码编写难度。
``` c++
class Solution {
public:
    bool isMatch(string s, string p) {
        int m = s.size();
        int n = p.size();
        vector<vector<int>> dp(m + 1, vector<int>(n + 1));
        dp[0][0] = true;
        for (int i = 1; i <= n; ++i) {
            if (p[i - 1] == '*') {
                dp[0][i] = true;
            }
            else {
                break;
            }
        }
        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (p[j - 1] == '*') {
                    dp[i][j] = dp[i][j - 1] | dp[i - 1][j];
                }
                else if (p[j - 1] == '?' || s[i - 1] == p[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1];
                }
            }
        }
        return dp[m][n];
    }
};
```

### 贪心算法
-动态规划的瓶颈在于对星号 $*$ 的处理方式：使用动态规划枚举所有的情况。由于星号是万能的，连续的多个星号和单个星号实际上是等价的。
- 以 $p=*abcd*$ 为例，$p$ 可以匹配**所有包含子串 $abcd$ 的字符串**，，也就是说，我们只需要暴力地枚举字符串 $s$ 中的每个位置作为起始位置，并判断对应的子串是否为 $abcd$ 即可。
- 如果 $p=∗ abcd∗efgh∗i ∗$ 呢？显然，$p$ 可以匹配所有依次出现子串$abcd$、$efgh$、$i$ 的字符串。因此，如果模式 $p$ 的形式为 $*~u_1*u_2*u_3~*\cdots*u_x~*$， 即字符串（可以为空）和星号交替出现，并且首尾字符均为星号，那么我们就可以设计出下面这个基于贪心的暴力匹配算法。`算法的本质是：`如果在字符串 $s$ 中首先找到 $u_1$，再找到 $u_2, u_3, \cdots, u_x$，那么 $s$ 就可以与模式 $p$ 匹配。
- 如果模式 $p$ 的结尾字符不是星号，那么就必须与字符串 $s$ 的结尾字符匹配。那么我们不断地匹配 $s$ 和 $p$ 的结尾字符，直到 $p$ 为空或者 $p$ 的结尾字符是星号为止。在这个过程中，如果匹配失败，或者最后 $p$ 为空但 $s$ 不为空，那么需要返回 $False$。
- 如果模式 $p$ 的开头字符不是星号，可以不断地匹配 $s$ 和 $p$ 的开头字符。
``` c++
class Solution {
public:
    bool isMatch(string s, string p) {
        // 如果 p 没有匹配完，那么 p 剩余的字符必须都是星号
        auto allStars = [](const string& str, int left, int right) {
            for (int i = left; i < right; ++i) {
                if (str[i] != '*') {
                    return false;
                }
            }
            return true;
        };
        auto charMatch = [](char u, char v) {
            return u == v || v == '?';
        };

        while (s.size() && p.size() && p.back() != '*') {
            if (charMatch(s.back(), p.back())) {
                s.pop_back();
                p.pop_back();
            }
            else {
                return false;
            }
        }
        if (p.empty()) {
            return s.empty();
        }

        int sIndex = 0, pIndex = 0;
        int sRecord = -1, pRecord = -1;
        while (sIndex < s.size() && pIndex < p.size()) {
            if (p[pIndex] == '*') {
                ++pIndex;
                sRecord = sIndex;
                pRecord = pIndex;
            }
            else if (charMatch(s[sIndex], p[pIndex])) {
                ++sIndex;
                ++pIndex;
            }
            else if (sRecord != -1 && sRecord + 1 < s.size()) {
                ++sRecord;
                sIndex = sRecord;
                pIndex = pRecord;
            }
            else {
                return false;
            }
        }
        return allStars(p, pIndex, p.size());
    }
};
```
## 跳跃游戏
### 反向查找
- 我们的目标是到达数组的最后一个位置，因此我们可以考虑最后一步跳跃前所在的位置，该位置通过跳跃能够到达最后一个位置。
- 我们可以**贪心**地选择距离最后一个位置最远的那个位置，也就是对应下标最小的那个位置。因此，我们可以从左到右遍历数组，选择第一个满足要求的位置。
- 找到最后一步跳跃前所在的位置之后，我们继续贪心地寻找倒数第二步跳跃前所在的位置，以此类推，直到找到数组的开始位置。
``` c++
class Solution {
    public int jump(vector<int> nums) {
        int position = nums.length - 1;
        int steps = 0;
        while (position > 0) {
            for (int i = 0; i < position; i++) {
                if (i + nums[i] >= position) {
                    position = i;
                    steps++;
                    break;
                }
            }
        }
        return steps;
    }
}
```
- 时间复杂度：$O(n^2)$。有两层嵌套循环，在最坏的情况下，例如数组中的所有元素都是 $1$，因此会超时。
### 正向查找
- 我们**贪心**地进行**正向查找**，每次找到可到达的最远位置，就可以在线性时间内得到最少的跳跃次数。
- 在具体的实现中，我们维护当前能够到达的最大下标位置，记为边界。我们从左到右遍历数组，到达边界时，更新边界并将跳跃次数增加 $1$。
- 在遍历数组时，我们不访问最后一个元素，这是因为在访问最后一个元素之前，我们的边界一定大于等于最后一个位置，否则就无法跳到最后一个位置了。如果访问最后一个元素，在边界正好为最后一个位置的情况下，我们会增加一次**不必要的跳跃次数**，因此我们不必访问最后一个元素。
<img src="https://assets.leetcode-cn.com/solution-static/45/45_fig1.png" width="600" height="400">
``` c++
class Solution {
public:
    int jump(vector<int>& nums) {
        int maxPos = 0, n = nums.size(), end = 0, step = 0;
        for (int i = 0; i < n - 1; ++i) {
            if (maxPos >= i) {
                maxPos = max(maxPos, i + nums[i]);
                if (i == end) {
                    end = maxPos;
                    ++step;
                }
            }
        }
        return step;
    }
};
```
## 全排列
回溯法：一种通过探索所有可能的候选解来找出所有的解的算法。如果候选解被确认不是一个解（或者至少不是最后一个解），回溯算法会通过在上一步进行一些变化抛弃该解，即回溯并且再次尝试。
### 思路
- 定义递归函数 $backtrack(first, output)$ 表示从左往右填到第 $first$ 个位置，当前排列为 $output$。 那么整个递归函数分为两个情况：
    + 如果 $first==n$，说明我们已经填完了 $n$ 个位置（注意下标从 $0$ 开始），找到了一个可行的解，我们将 $output$ 放入答案数组中，递归结束。
    + 如果 $first<n$，我们要考虑这第 $first$ 个位置我们要填哪个数。根据题目要求我们肯定不能填已经填过的数，因此很容易想到的一个处理手段是我们定义一个标记数组 $vis[]$ 来标记已经填过的数，那么在填第 $first$ 个数的时候我们遍历题目给定的 $n$ 个数，如果这个数没有被标记过，我们就尝试填入，并将其标记，继续尝试填下一个位置，即调用函数 $backtrack(first + 1, output)$。回溯的时候要撤销这一个位置填的数以及标记，并继续尝试其他没被标记过的数。
### 解法
- 使用标记数组来处理填过的数是一个很直观的思路，但增加了算法的空间复杂度。可以将题目给定的 $n$ 个数的数组 $nums$ 划分成左右两个部分，左边的表示已经填过的数，右边表示待填的数，我们在回溯的时候只要动态维护这个数组即可。
- 具体来说，假设我们已经填到第 $first$ 个位置，那么 $nums$ 数组中 $[0,first−1]$ 是已填过的数的集合，$[first,n−1]$ 是待填的数的集合。我们肯定是尝试用 $[first,n−1]$ 里的数去填第 $first$ 个数，假设待填的数的下标为 $i$ ，那么填完以后我们将第 $i$ 个数和第 $first$ 个数交换，即能使得在填第 $first+1$ 个数的时候 $nums$ 数组的 $[0,first]$ 部分为已填过的数，$[first+1,n−1]$ 为待填的数，回溯的时候交换回来即能完成撤销操作。
- **注意**：这样生成的全排列并不是按字典序存储在答案数组中的，如果题目要求按字典序输出，那么请还是用标记数组或者其他方法。
<img src="https://assets.leetcode-cn.com/solution-static/46/fig14.PNG" width="600" height="400">
``` c++
class Solution {
public:
    void backtrack(vector<vector<int>>& res, vector<int>& output, int first, int len){
        // 所有数都填完了
        if (first == len) {
            res.emplace_back(output);
            return;
        }
        for (int i = first; i < len; ++i) {
            // 动态维护数组
            swap(output[i], output[first]);
            // 继续递归填下一个数
            backtrack(res, output, first + 1, len);
            // 撤销操作
            swap(output[i], output[first]);
        }
    }
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int> > res;
        backtrack(res, nums, 0, (int)nums.size());
        return res;
    }
};
```
- 时间复杂度：$O(n \times n!)$
## 全排列2
### 思想
- 总体思路同上，但需要满足**全排列不重复**的要求；
- 要解决重复问题，我们只要设定一个规则，保证在填第 $idx$ 个数的时候重复数字只会被填入一次即可。而在本题解中，我们选择对原数组排序，保证相同的数字都相邻，然后每次填入的数一定是这个数所在重复数集合中**从左往右第一个未被填过的数字**，即如下的判断条件：
``` c++
if (i > 0 && nums[i] == nums[i - 1] && !vis[i - 1]) {
    continue;
}
```
- 这个判断条件保证了对于重复数的集合，一定是从左往右逐个填入的。
### 分析
- 画出树形结构如下：重点想象深度优先遍历在这棵树上执行的过程，哪些地方遍历下去一定会产生重复，这些地方的状态的特点是什么
<img src="https://pic.leetcode-cn.com/1600386643-uhkGmW-image.png" width="800" height="500">
- 产生重复结点的地方，正是图中标注了「剪刀」，且被红色框框住的地方。
- 在图中 ① 处，搜索的数也和上一次一样，但是上一次的 1 刚刚被撤销，正是因为刚被撤销，下面的搜索中还会使用到，因此会产生重复，剪掉的就应该是这样的分支。
``` c++
class Solution {
    vector<int> vis;

public:
    void backtrack(vector<int>& nums, vector<vector<int>>& ans, int idx, vector<int>& perm) {
        if (idx == nums.size()) {
            ans.emplace_back(perm);
            return;
        }
        for (int i = 0; i < (int)nums.size(); ++i) {
            if (vis[i] || (i > 0 && nums[i] == nums[i - 1] && !vis[i - 1])) {
                continue;
            }
            perm.emplace_back(nums[i]);
            vis[i] = 1;
            backtrack(nums, ans, idx + 1, perm);
            vis[i] = 0;
            perm.pop_back();
        }
    }

    vector<vector<int>> permuteUnique(vector<int>& nums) {
        vector<vector<int>> ans;
        vector<int> perm;
        vis.resize(nums.size());
        sort(nums.begin(), nums.end());
        backtrack(nums, ans, 0, perm);
        return ans;
    }
};
```
## 旋转图像
### 原地旋转
> 矩阵中第 $i$ 行的第 $j$ 个元素，在旋转后，会出现在倒数第 $i$ 列的第 $j$ 个位置。

即有：
$$
\begin{cases} matrix[col][n-row-1] & = matrix[row][col]\\
matrix[n-row-1][n-col-1] & = matrix[col][n-row-1]\\
matrix[n-col-1][row] & = matrix[n-row-1][n-col-1]\\
matrix[row][col] & = matrix[n-col-1][row]\end{cases}
$$
当我们知道了如何原地旋转矩阵之后，还有一个重要的问题在于：我们应该枚举哪些位置 $(row,col)$ 进行上述的原地交换操作呢？由于每一次原地交换四个位置，因此：
- 当 $n$ 为偶数时，我们需要枚举 $n^2 / 4 = (n/2) \times (n/2)$ 个位置，可以将该图形分为四块，以 $4 \times 4$ 的矩阵为例：
<img src="https://assets.leetcode-cn.com/solution-static/48/1.png" width="700" height="400">
- 当 $n$ 为奇数时，由于中心的位置经过旋转后位置不变，我们需要枚举 $(n^2-1) / 4 = ((n-1)/2) \times ((n+1)/2)$ 个位置，需要换一种划分的方式，以 $5 \times 5$的矩阵为例：
<img src="https://assets.leetcode-cn.com/solution-static/48/2.png" width="700" height="400">
``` c++
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int n = matrix.size();
        for (int i = 0; i < n / 2; ++i) {
            for (int j = 0; j < (n + 1) / 2; ++j) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[n - j - 1][i];
                matrix[n - j - 1][i] = matrix[n - i - 1][n - j - 1];
                matrix[n - i - 1][n - j - 1] = matrix[j][n - i - 1];
                matrix[j][n - i - 1] = temp;
            }
        }
    }
};
```
### 翻转代替旋转
先将矩阵沿水平轴翻转：
$$
\left[ \begin{matrix} 5 & 1 & 9 & 11 \\ 2 & 4 & 8 & 10 \\ 13 & 3 & 6 & 7 \\ 15 & 14 & 12 & 16 \end{matrix} \right] \Longrightarrow{水平} \left[ \begin{matrix} 15 & 14 & 12 & 16 \\ 13 & 3 & 6 & 7 \\ 2 & 4 & 8 & 10 \\  5 & 1 & 9 & 11  \end{matrix} \right]
$$
再根据主对角线翻转得到：
$$
\left[ \begin{matrix} 15 & 14 & 12 & 16 \\ 13 & 3 & 6 & 7 \\ 2 & 4 & 8 & 10 \\  5 & 1 & 9 & 11  \end{matrix} \right] \Longrightarrow{主对角线} \left[ \begin{matrix} 15 & 13 & 2 & 5 \\ 14 & 3 & 4 & 1 \\ 12 & 6 & 8 & 9 \\ 16 & 7 & 10 & 11 \end{matrix} \right]
$$
对于水平轴翻转而言，我们只需要枚举矩阵上半部分的元素，和下半部分的元素进行交换，即：
$$
matrix[row][col] \Longrightarrow{水平} matrix[n-row-1][col]
$$
对于主对角线翻转而言，我们只需要枚举对角线左侧的元素，和右侧的元素进行交换，即：
$$
matrix[row][col] \Longrightarrow{主对角线} matrix[col][row]
$$
将它们联立即可得到：
$$
matrix[row][col] \Longrightarrow{水平} matrix[n-row-1][col]  \Longrightarrow{主对角线} matrix[col][n-row-1]
$$
``` c++
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int n = matrix.size();
        // 水平翻转
        for (int i = 0; i < n / 2; ++i) {
            for (int j = 0; j < n; ++j) {
                swap(matrix[i][j], matrix[n - i - 1][j]);
            }
        }
        // 主对角线翻转
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                swap(matrix[i][j], matrix[j][i]);
            }
        }
    }
};
```
## 字母异位词分组
两个字符串互为字母异位词，当且仅当两个字符串包含的字母相同。同一组字母异位词中的字符串具备相同点，可以使用相同点作为一组字母异位词的标志，使用哈希表存储每一组字母异位词，哈希表的键为一组字母异位词的标志，哈希表的值为一组字母异位词列表。
遍历每个字符串，对于每个字符串，得到该字符串所在的一组字母异位词的标志，将当前字符串加入该组字母异位词的列表中。遍历全部字符串之后，哈希表中的每个键值对即为一组字母异位词。
### 排序
由于互为字母异位词的两个字符串包含的字母相同，因此对两个字符串分别进行排序之后得到的字符串一定是相同的，故可以将排序之后的字符串作为哈希表的键。
``` c++
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        unordered_map<string, vector<string>> mp;
        for (string& str: strs) {
            string key = str;
            sort(key.begin(), key.end());
            mp[key].emplace_back(str);
        }
        vector<vector<string>> ans;
        for (auto it = mp.begin(); it != mp.end(); ++it) {
            ans.emplace_back(it->second);
        }
        return ans;
    }
};
```
- 时间复杂度：$O(nk logk)$。其中 $n$ 是 $strs$ 中的字符串的数量，$k$ 是 $strs$ 中的字符串的的最大长度。
### 计数
由于互为字母异位词的两个字符串包含的字母相同，因此两个字符串中的相同字母出现的次数一定是相同的，故可以将每个字母出现的次数使用字符串表示，作为哈希表的键。
由于字符串只包含小写字母，因此对于每个字符串，可以使用长度为 $26$ 的数组记录每个字母出现的次数。需要注意的是，在使用数组作为哈希表的键时，不同语言的支持程度不同，因此不同语言的实现方式也不同。
``` c++
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        // 自定义对 array<int, 26> 类型的哈希函数
        auto arrayHash = [fn = hash<int>{}] (const array<int, 26>& arr) -> size_t {
            return accumulate(arr.begin(), arr.end(), 0u, [&](size_t acc, int num) {
                return (acc << 1) ^ fn(num);
            });
        };

        unordered_map<array<int, 26>, vector<string>, decltype(arrayHash)> mp(0, arrayHash);
        for (string& str: strs) {
            array<int, 26> counts{};
            int length = str.length();
            for (int i = 0; i < length; ++i) {
                counts[str[i] - 'a'] ++;
            }
            mp[counts].emplace_back(str);
        }
        vector<vector<string>> ans;
        for (auto it = mp.begin(); it != mp.end(); ++it) {
            ans.emplace_back(it->second);
        }
        return ans;
    }
};
```
- 时间复杂度：$O(n(k+|\Sigma|))$