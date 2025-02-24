'''
347. Top K Frequent Elements
Medium
Given an integer array nums and an integer k, return the k most frequent elements. You may return the answer in any order.

Example 1:
Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]

Example 2:
Input: nums = [1], k = 1
Output: [1]
 
Constraints:
* 1 <= nums.length <= 10^5
* -10^4 <= nums[i] <= 10^4
* k is in the range [1, the number of unique elements in the array].
* It is guaranteed that the answer is unique.
'''

# Core Concepts: Hash maps (dictionaries) for counting frequency, sorting, and heaps (priority queues).
from collections import Counter

def topKFrequent(nums: List[int], k: int) -> List[int]:
    # Count the frequency of each element
    counts = Counter(nums)

    # Sort elements by frequency (descending)
    counts_list = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    # Extract the top k elements
    sorted_counts = dict(counts_list[:k])

    return [num for num in sorted_counts]
