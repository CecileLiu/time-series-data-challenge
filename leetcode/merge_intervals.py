'''
56. Merge Intervals
Medium
Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.


Example 1:
Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlap, merge them into [1,6].

Example 2:
Input: intervals = [[1,4],[4,5]]
Output: [[1,5]]
Explanation: Intervals [1,4] and [4,5] are considered overlapping.
 

Constraints:
* 1 <= intervals.length <= 10^4
* intervals[i].length == 2
* 0 <= start_i <= end_i <= 10^4
'''

class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        if not intervals:
            return intervals
        
        # 對區間列表進行排序
        intervals.sort(key=lambda x: x.start)
        # 準備一個容器來裝合併的元素 (索引 0 先放進去)
        result = [intervals[0]]
        # 原本的陣列從第二個開始循環比較
        # intervals[1:]
        # Iteration 1: start = 2, end = 6
        # Iteration 2: start = 8, end = 10
        # Iteration 3: start = 15, end = 18
        for i in xrange(1, len(intervals)):
            prev, current = result[-1], intervals[i]
            if current.start <= prev.end: 
                prev.end = max(prev.end, current.end)
            else:
                result.append(current)
        return result


'''Reference
https://github.com/ChenglongChen/LeetCode-3/blob/master/Python/merge-intervals.py
https://medium.com/@aien1020210305/leetcode-%E7%AD%86%E8%A8%98-56-merge-intervals-324ae5c108c4
'''