## 寻找旋转排序数组中的最小值

### 二分查找

一个不包含重复元素的升序数组在经过旋转之后，可以得到下面可视化的折线图：

<img src="https://assets.leetcode-cn.com/solution-static/153/1.png" width="800" >

考虑数组中的最后一个元素 $x$：在最小值右侧的元素（不包括最后一个元素本身），它们的值一定都严格小于 $x$；而在最小值左侧的元素，它们的值一定都严格大于 $x$​。因此，我们可以根据这一条性质，通过二分查找的方法找出最小值：

在二分查找的每一步中，左边界为 $\it low$，右边界为 $\it high$，区间的中点为 $\it pivot$，最小值就在该区间内。我们将中轴元素 $\textit{nums}[\textit{pivot}]$ 与右边界元素 $\textit{nums}[\textit{high}]$​ 进行比较，可能会有以下的三种情况：

- 第一种情况是 $\textit{nums}[\textit{pivot}] < \textit{nums}[\textit{high}]$。这说明 $\textit{nums}[\textit{pivot}]$​​ 是最小值右侧的元素，因此我们可以忽略二分查找区间的右半部分。
- 第二种情况是 $\textit{nums}[\textit{pivot}] > \textit{nums}[\textit{high}]$。如下图所示，这说明 $\textit{nums}[\textit{pivot}]$ 是最小值左侧的元素，因此我们可以忽略二分查找区间的左半部分。

- 由于数组不包含重复元素，并且只要当前的区间长度不为 $1$，$\it pivot$ 就不会与 $\it high$ 重合；而如果当前的区间长度为 $1$，这说明我们已经可以结束二分查找了。因此不会存在 $\textit{nums}[\textit{pivot}] = \textit{nums}[\textit{high}]$​ 的情况。

当二分查找结束时，我们就得到了最小值所在的位置：

``` c++
class Solution {
public:
    int findMin(vector<int>& nums) {
        int low = 0;
        int high = nums.size() - 1;
        while (low < high) {
            int pivot = low + (high - low) / 2;
            if (nums[pivot] < nums[high]) {
                high = pivot;
            }
            else {
                low = pivot + 1;
            }
        }
        return nums[low];
    }
};
```

## 寻找旋转排序数组中的最小值2（存在重复元素）

### 二分查找

一个包含重复元素的升序数组在经过旋转之后，可以得到下面可视化的折线图：

<img src="https://assets.leetcode-cn.com/solution-static/154/1.png" width="800">



考虑数组中的最后一个元素 $x$：在最小值右侧的元素（不包括最后一个元素本身），它们的值一定都小于等于 $x$；而在最小值左侧的元素，它们的值一定都大于等于 $x$​。因此，我们可以根据这一条性质，通过二分查找的方法找出最小值：

在二分查找的每一步中，左边界为 $\it low$，右边界为 $\it high$，区间的中点为 $\it pivot$，最小值就在该区间内。我们将中轴元素 $\textit{nums}[\textit{pivot}]$ 与右边界元素 $\textit{nums}[\textit{high}]$​ 进行比较，可能会有以下的三种情况：

- 第一种情况是 $\textit{nums}[\textit{pivot}] < \textit{nums}[\textit{high}]$。这说明 $\textit{nums}[\textit{pivot}]$​​ 是最小值右侧的元素，因此我们可以忽略二分查找区间的右半部分。
- 第二种情况是 $\textit{nums}[\textit{pivot}] > \textit{nums}[\textit{high}]$。如下图所示，这说明 $\textit{nums}[\textit{pivot}]$ 是最小值左侧的元素，因此我们可以忽略二分查找区间的左半部分。
- 第三种情况是 $\textit{nums}[\textit{pivot}] == \textit{nums}[\textit{high}]$。如下图所示，由于重复元素的存在，我们并不能确定 $\textit{nums}[\textit{pivot}]$ 究竟在最小值的左侧还是右侧，因此我们不能莽撞地忽略某一部分的元素。我们唯一可以知道的是，由于它们的值相同，所以无论 $\textit{nums}[\textit{high}]$ 是不是最小值，都有一个它的「替代品」$\textit{nums}[\textit{pivot}]$​，因此我们可以忽略二分查找区间的右端点。

<img src="https://assets.leetcode-cn.com/solution-static/154/4.png" width="800">

``` c++
class Solution {
public:
    int findMin(vector<int>& nums) {
        int low = 0;
        int high = nums.size() - 1;
        while (low < high) {
            int pivot = low + (high - low) / 2;
            if (nums[pivot] < nums[high]) {
                high = pivot;
            }
            else if (nums[pivot] > nums[high]) {
                low = pivot + 1;
            }
            else {
                high -= 1;
            }
        }
        return nums[low];
    }
};
```



