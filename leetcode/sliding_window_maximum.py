''' 
239. Sliding Window Maximum
Hard
You are given an array of integers nums, there is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window. Each time the sliding window moves right by one position.

Return the max sliding window.

Example 1:
Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [3,3,5,5,6,7]
Explanation: 
Window position                Max
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7

Example 2:
Input: nums = [1], k = 1
Output: [1]
 

Constraints:
* 1 <= nums.length <= 10^5
* -10^4 <= nums[i] <= 10^4
* 1 <= k <= nums.length
'''

class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        # Edge case: if nums is empty or k is 0, return an empty list
        if not nums or k == 0:
            return []

        max_numbers = []  # This will store the maximums for each window
        dq = deque()  # Deque to store indices of elements in the current window

        for i in range(len(nums)):
            # Remove elements that are smaller than the current element 
            while dq and nums[i] >= nums[dq[-1]]: # 2. 如果下一個比現在這一個大，就移除前一個加進來的 # pop element從後面pop
                dq.pop()
            dq.append(i) # !. 不管如何，就是從後面加element
            # Remove indices that are out of this window
            if i >= k and dq and dq[0] <= i - k: # 4. 因為要確保deque最左邊的值都在每一個window範圍，所以當新一輪的window啟動，就要來檢查一下
                dq.popleft()
            # 當i已經走到window最右邊的值
            if i >= k - 1:
                 max_numbers.append(nums[dq[0]]) # 3. 經過while和dq.append的運作，已經確保了第0位置的值是這一個window最大

'''Reference
answer from Perplexity
https://github.com/ChenglongChen/LeetCode-3/blob/master/Python/sliding-window-maximum.py
'''