import sun.awt.image.ImageWatched;

import javax.swing.*;
import java.util.*;
import java.lang.*;

public class workout {
    private Object WordDictionary;

    public int longestIncreasingPath(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        boolean[][] visited = new boolean[m][n];
        int[][] memo = new int[m][n];
        int ans = -1;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                ans = Math.max(backtrack(matrix, i, j, m, n, visited, memo), ans);
            }
        }
        return ans;
    }

    int backtrack(int[][] matrix, int x, int y, int m, int n, boolean[][] visited, int[][] memo) {
        if (memo[x][y] != 0) {
            return memo[x][y];
        }
        memo[x][y]++;
        visited[x][y] = true;
        for (int i = 0; i < 4; i++) {
            int nx = x + dx[i];
            int ny = y + dy[i];
            if (nx >= 0 && nx < m && ny >= 0 && ny < n && !visited[nx][ny] && matrix[nx][ny] > matrix[x][y]) {
                memo[x][y] = Math.max(memo[x][y], backtrack(matrix, nx, ny, m, n, visited, memo) + 1);
            }
        }
        visited[x][y] = false;
        return memo[x][y];
    }

    int[] dx = {1, 0, -1, 0}, dy = {0, 1, 0, -1};

    HashMap<TreeNode, Integer> mp = new HashMap<>();

    public int maxPathSum(TreeNode root) {
        return root.val;
    }

    int traverse(TreeNode root) {
        if (mp.containsKey(root)) {
            return mp.get(root);
        }
        if (root == null) {
            return 0;
        }
        int leftMax = Integer.MIN_VALUE, rightMax = Integer.MIN_VALUE;
        if (root.left != null) {
            leftMax = traverse(root.left);
        }
        if (root.right != null) {
            rightMax = traverse(root.right);
        }
        return root.val;
    }

    public int n, k;
    public int[] fac;
    public boolean[] used;

    public String getPermutation(int n, int k) {
        this.n = n;
        this.k = k;
        fac = new int[n + 1];
        used = new boolean[n + 1];
        calcFac();
        StringBuilder ans = new StringBuilder();
        dfs(ans, 0);
        return ans.toString();
    }

    void dfs(StringBuilder ans, int index) {
        if (index == n) {
            return;
        }
        int cnt = fac[n - 1 - index];
        for (int i = 1; i <= n; i++) {
            if (used[i]) {
                continue;
            }
            if (cnt < k) {
                k -= cnt;
                continue;
            }
            ans.append(i);
            used[i] = true;
            dfs(ans, index + 1);
            return;
        }

    }

    void calcFac() {
        fac = new int[n + 1];
        fac[0] = 1;
        for (int i = 1; i < fac.length; i++) {
            fac[i] = fac[i - 1] * i;
        }
    }

    public boolean isMatch(String s, String p) {
        int m = s.length(), n = p.length();
        boolean[][] dp = new boolean[m + 1][n + 1];
        for (int i = 0; i <= m; i++) {
            dp[i][0] = false;
        }
        dp[0][0] = true;
        if (isNullMatched(p)) {
            for (int i = 0; i <= n; i += 2) {
                dp[0][i] = true;
            }
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (j - 1 >= 0 && p.charAt(j) != '*') {
                    if (s.charAt(i) == p.charAt(j - 1) || p.charAt(j - 1) == '.') {
                        dp[i + 1][j + 1] |= dp[i][j];
                    } else {
                        dp[i + 1][j + 1] = false;
                    }
                } else if (j - 1 >= 0 && p.charAt(j) == '*') {
                    if (s.charAt(i) == p.charAt(j - 1) || p.charAt(j - 1) == '.') {
                        dp[i + 1][j + 1] |= dp[i][j - 1] | dp[i][j + 1] | dp[i + 1][j - 1];
                    } else {
                        dp[i + 1][j + 1] |= dp[i + 1][j - 1];
                    }
                }
            }
        }
        return dp[m][n];
    }

    public boolean isNullMatched(String s) {
        int n = s.length();
        if (n % 2 != 0) {
            return false;
        }
        for (int i = 1; i < n; i += 2) {
            if (s.charAt(i) != '*') {
                return false;
            }
        }
        return true;
    }

    public int getKthMagicNumber(int k) {
        int[] dp = new int[k];
        int p1 = 0, p2 = 0, p3 = 0;
        dp[0] = 1;
        for (int i = 1; i < k; i++) {
            dp[i] = Math.min(dp[p1] * 3, Math.min(dp[p2] * 5, dp[p3] * 7));
            if (dp[i] == dp[p1] * 3) p1++;
            if (dp[i] == dp[p2] * 5) p2++;
            if (dp[i] == dp[p3] * 7) p3++;
        }
        return dp[k - 1];
    }

    public List<List<String>> groupAnagrams(String[] strs) {
        HashMap<String, List<String>> mp = new HashMap<>();
        for (String str : strs) {
            char[] arr = str.toCharArray();
            Arrays.sort(arr);
            String temp = Arrays.toString(arr);
            if (mp.containsKey(temp)) {
                mp.get(temp).add(str);
            } else {
                mp.put(temp, new LinkedList<>());
                mp.get(temp).add(str);
            }
        }
        List<List<String>> ans = new LinkedList<>();
        for (String s : mp.keySet()) {
            ans.add(mp.get(s));
        }
        return ans;
    }

    public List<String> wordBreak(String s, List<String> wordDict) {
        HashSet<String> set = new HashSet<>(wordDict);
        int n = s.length();
        boolean[] dp = new boolean[n + 1];
        dp[0] = true;
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < i; j++) {
                dp[i] = dp[j] & set.contains(s.substring(j, i));
                if (dp[i]) {
                    break;
                }
            }
        }
        backtrack(s, dp, set, 0, n);
        return ans;
    }

    List<String> ans = new LinkedList<>();
    StringBuilder track = new StringBuilder();

    void backtrack(String s, boolean[] dp, HashSet<String> wordList, int cur, int n) {
        if (cur == n) {
            track.deleteCharAt(track.length() - 1);
            ans.add(track.toString());
            track.append(" ");
            return;
        }
        for (int i = cur + 1; i <= n; i++) {
            if (dp[i] && wordList.contains(s.substring(cur, i))) {
                track.append(s, cur, i);
                track.append(" ");
                backtrack(s, dp, wordList, i, n);
                track.deleteCharAt(track.length() - 1);
                track.delete(track.length() - (i - cur), track.length());
            }
        }
    }

    public int[][] merge(int[][] intervals) {
        Arrays.sort(intervals, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] - o2[0];
            }
        });
        int n = intervals.length;
        LinkedList<LinkedList<Integer>> ans = new LinkedList<>();
        ans.add(new LinkedList<>());
        ans.get(0).add(intervals[0][0]);
        ans.get(0).add(intervals[0][1]);
        for (int i = 1; i < n; i++) {
            if (intervals[i][0] <= ans.getLast().get(1) && intervals[i][1] > ans.getLast().get(1)) {
                ans.getLast().removeLast();
                ans.getLast().add(intervals[i][1]);
            } else if (intervals[i][0] > ans.getLast().get(1)) {
                ans.add(new LinkedList<>());
                ans.getLast().add(intervals[i][0]);
                ans.getLast().add(intervals[i][1]);
            }
        }
        int[][] res = new int[ans.size()][2];
        for (int i = 0; i < ans.size(); i++) {
            res[i][0] = ans.get(i).get(0);
            res[i][1] = ans.get(i).get(1);
        }
        return res;
    }


    int Paritition1(int[] A, int low, int high) {
        int pivot = A[low];
        while (low < high) {
            while (low < high && A[high] >= pivot) {
                --high;
            }
            A[low] = A[high];
            while (low < high && A[low] <= pivot) {
                ++low;
            }
            A[high] = A[low];
        }
        A[low] = pivot;
        return low;
    }

    void QuickSort(int[] A, int low, int high) //快排母函数
    {
        if (low < high) {
            int pivot = Paritition1(A, low, high);
            QuickSort(A, low, pivot - 1);
            QuickSort(A, pivot + 1, high);
        }
    }

    public int findArray(int[] arr) {//记1比0多的数目
        int n = arr.length;
        int[][] dp = new int[n][n];
        for (int i = 0; i < n; i++) {
            dp[i][i] = arr[i] == 1 ? 1 : -1;
        }
        int max = -1;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                dp[i][j] = dp[i][j - 1] + (arr[j] == 1 ? 1 : -1);
                if (dp[i][j] == 0) {
                    max = Math.max(j - i + 1, max);
                }
            }
        }
        return max;
    }

    public int solution(int[] A) {
        // write your code in Java SE 8
        int maxValue = 1000010;
        boolean[] table = new boolean[maxValue];
        Arrays.sort(A);
        int n = A.length;
        for (int i = 0; i < n; i++) {
            if (A[i] > 0) {
                table[A[i]] = true;
            }
        }
        for (int i = 1; i < maxValue; i++) {
            if (!table[i]) {
                return i;
            }
        }
        return 1;
    }

    public boolean canFinish(int numCourses, int[][] prerequisites) {
        int[] indegrees = new int[numCourses];
        for (int[] pre : prerequisites) {
            indegrees[pre[0]]++;
        }
        Queue<Integer> q = new LinkedList<>();
        for (int i = 0; i < indegrees.length; i++) {
            if (indegrees[i] == 0) {
                q.offer(i);
            }
        }
        while (!q.isEmpty()) {
            int sz = q.size();
            for (int i = 0; i < sz; i++) {
                int cur = q.poll();
                for (int[] pre : prerequisites) {
                    if (pre[1] == cur) {
                        indegrees[pre[0]]--;
                        if (indegrees[pre[0]] == 0) {
                            q.offer(pre[0]);
                        }
                    }
                }
                numCourses--;
            }
        }
        return numCourses == 0;
    }

    public int[] findOrder(int numCourses, int[][] prerequisites) {
        int[] indegrees = new int[numCourses];
        for (int[] pre : prerequisites) {
            indegrees[pre[0]]++;
        }
        int[] ans = new int[numCourses];
        int start = 0;
        Queue<Integer> q = new LinkedList<>();
        for (int i = 0; i < numCourses; i++) {
            if (indegrees[i] == 0) {
                q.offer(i);
            }
        }
        while (!q.isEmpty()) {
            int sz = q.size();
            for (int i = 0; i < sz; i++) {
                int cur = q.poll();
                for (int[] pre : prerequisites) {
                    if (pre[1] == cur) {
                        indegrees[pre[0]]--;
                        if (indegrees[pre[0]] == 0) {
                            q.offer(pre[0]);
                        }
                    }
                }
                ans[start++] = cur;
            }
        }
        if (start != numCourses) {
            return new int[]{};
        }
        return ans;
    }

    public int reverseBits(int n) {
        int result = 0;
        for (int i = 0; i < 32; i++) {
            result <<= 1;
            result += 1 & n;
            n >>= 1;
        }
        return result;
    }

    public int longestSubstring(String s, int k) {
        int n = s.length();
        int maxLen = 0;
        for (int t = 1; t <= 26; t++) {
            int left = 0, right = 0;
            int notValid = 0;
            int total = 0;
            int[] count = new int[26];
            while (right < n) {
                char ch = s.charAt(right++);
                count[ch - 'a']++;
                if (count[ch - 'a'] == 1) {
                    notValid++;
                    total++;
                }
                if (count[ch - 'a'] == k) {
                    notValid--;
                }
                while (total > t) {
                    char c = s.charAt(left++);
                    count[c - 'a']--;
                    if (count[c - 'a'] == k - 1) {
                        notValid++;
                    }
                    if (count[c - 'a'] == 0) {
                        total--;
                        notValid--;
                    }
                }
                if (notValid == 0) {
                    maxLen = Math.max(maxLen, right - left);
                }
            }
        }
        return maxLen;
    }

    public String countAndSay(int n) {
        if (n == 1) {
            return "1";
        }
        String s = "1";
        for (int i = 0; i < n - 1; i++) {
            s = nextString(s);
        }
        return s;
    }

    public String nextString(String s) {
        int n = s.length();
        char c = s.charAt(0);
        int cnt = 1;
        String ans = "";
        if (n == 1) {
            return "11";
        }
        for (int i = 1; i < n; i++) {
            if (s.charAt(i) == c) {
                cnt++;
                if (i == n - 1) {
                    ans = ans + cnt + c;
                }
            } else {
                ans = ans + cnt + c;
                c = s.charAt(i);
                cnt = 1;
                if (i == n - 1) {
                    ans = ans + cnt + c;
                }
            }
        }
        return ans;
    }

    public int[] shortestSeq(int[] big, int[] small) {
        HashSet<Integer> need = new HashSet<>();
        HashMap<Integer, Integer> window = new HashMap<>();
        for (int i : small) {
            need.add(i);
        }
        int n = big.length;
        int valid = 0;
        int left = 0, right = 0;
        int minLen = Integer.MAX_VALUE;
        int start = -1;
        while (right < n) {
            int num = big[right++];
            window.put(num, window.getOrDefault(num, 0) + 1);
            if (window.get(num) == 1 && need.contains(num)) {
                valid++;
            }
            while (valid == small.length) {
                if (right - left < minLen) {
                    start = left;
                    minLen = right - left;
                }
                int num1 = big[left++];
                window.put(num1, window.get(num1) - 1);
                if (window.get(num1) == 0 && need.contains(num1)) {
                    valid--;
                }
            }
        }
        if (start == -1) {
            return new int[]{};
        }
        return new int[]{start, start + minLen - 1};
    }

    public int minOperations(int[] nums, int x) {
        int sum = 0;
        for (int n : nums) {
            sum += n;
        }
        int target = sum - x;
        if (target < 0) {
            return -1;
        }
        if (target == 0) {
            return nums.length;
        }
        int left = 0, right = 0;
        int winSum = 0;
        int n = nums.length;
        int ans = Integer.MAX_VALUE;
        while (right < n) {
            int cur = nums[right++];
            winSum += cur;
            while (winSum >= target) {
                if (winSum == target) {
                    ans = Math.min(ans, left + (n - right));
                }
                winSum -= nums[left];
                if (left < n - 1) {
                    left++;
                }
            }
        }
        return ans == Integer.MAX_VALUE ? -1 : ans;
    }

    public int solution1(int[] A) {
        // write your code in Java SE 8
        HashMap<Integer, Integer> mp = new HashMap<>();
        int maxValue = Integer.MIN_VALUE;
        for (int a : A) {
            maxValue = Math.max(maxValue, a);
            mp.put(a, mp.getOrDefault(a, 0) + 1);
        }
        int ans = 0;
        for (int key : mp.keySet()) {
            if (key == maxValue) {
                ans += 1;
            } else {
                if (mp.get(key) >= 2) {
                    ans += 2;
                } else {
                    ans += 1;
                }
            }
        }
        return ans;
    }

    public int solution2(String L1, String L2) {
        // write your code in Java SE 8
        int N = L1.length();
        int L1total = 0, L2total = 0;
        for (int i = 0; i < N; i++) {
            if (L1.charAt(i) == 'x') {
                L1total++;
            }
            if (L2.charAt(i) == 'x') {
                L2total++;
            }
        }
        int[][] dp = new int[2][N];
        dp[0][0] = 0;
        dp[1][0] = 0;
        for (int i = 1; i < N; i++) {
            if (L1.charAt(i - 1) == 'x') {
                dp[0][i] = dp[0][i - 1] + 1;
            } else {
                dp[0][i] = dp[0][i - 1];
            }
            if (L2.charAt(i - 1) == 'x') {
                dp[1][i] = dp[1][i - 1] + 1;
            } else {
                dp[1][i] = dp[1][i - 1];
            }
        }
        int maxValue = Math.max(L1total, L2total);
        for (int i = 1; i < N - 1; i++) {
            maxValue = getMax(maxValue, dp[0][i] + (L2total - dp[1][i] - (L2.charAt(i) == 'x' ? 1 : 0)), dp[1][i] + (L1total - dp[0][i] - (L1.charAt(i) == 'x' ? 1 : 0)));
        }
        return maxValue;
    }

    int getMax(int a, int b, int c) {
        return Math.max(a, Math.max(b, c));
    }

    String binaryAdd(String s1, String s2) {
        int n1 = s1.length();
        int n2 = s2.length();
        int n = Math.max(n1, n2);
        String ans = "";
        s1 = reverse(s1);
        s2 = reverse(s2);
        int up = 0;
        for (int i = 0; i < n; i++) {

            int temp = s1.charAt(i) - '0' + s2.charAt(i) - '0' + up;
            System.out.println("first " + temp);
            if (temp == 0 || temp == 1) {
                ans = ans + (char) ('0' + temp);
                up = 0;
            } else {
                ans = ans + (char) ('0' + temp % 2);
                up = 1;
            }
        }
        return reverse(ans);
    }

    String reverse(String s) {
        int n = s.length();
        String ans = "";
        for (int i = n - 1; i >= 0; i--) {
            ans = ans + s.charAt(i);
        }
        return ans;
    }

    public int solution3(int[] A) {
        int[] table = new int[10010];
        int maxValue = -1;
        for (int a : A) {
            table[a]++;
            maxValue = Math.max(maxValue, a);
        }
        int ans = 0;
        for (int i = 0; i < 10009; i++) {
            if (table[i] == 0) {
                continue;
            }
            int up = table[i] / 2;
            table[i] = table[i] % 2;
            if (table[i] == 1) {
                ans++;
            }
            table[i + 1] += up;
        }
        return ans;
    }

    public List<Integer> spiralOrder(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
        boolean[][] visited = new boolean[m][n];
        List<Integer> ans = new LinkedList<>();
        int curX = 0, curY = 0, curDire = 0;
        while (!isEnd(matrix, curX, curY, m, n, visited)) {
            while (!isKnocked(matrix, curX + dire[curDire][0], curY + dire[curDire][1], m, n, visited)) {
                ans.add(matrix[curX][curY]);
                visited[curX][curY] = true;
                curX += dire[curDire][0];
                curY += dire[curDire][1];
            }
            curDire = (curDire + 1) % 4;
        }
        ans.add(matrix[curX][curY]);
        return ans;
    }

    int[][] dire = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

    boolean isKnocked(int[][] matrix, int x, int y, int m, int n, boolean[][] visited) {
        if (x == -1 || y == -1 || x == m || y == n || visited[x][y]) {
            return true;
        }
        return false;
    }

    boolean isEnd(int[][] matrix, int x, int y, int m, int n, boolean[][] visited) {
        for (int i = 0; i < 4; i++) {
            if (x + dire[i][0] >= 0 && x + dire[i][0] < m && y + dire[i][1] >= 0 && y + dire[i][1] < n && !visited[x + dire[i][0]][y + dire[i][1]]) {
                return false;
            }
        }
        return true;
    }

    public int maxLength(int[] arr) {
        // write code here
        int n = arr.length;
        int left = 0, right = 0;
        int maxLen = -1;
        LinkedList<Integer> window = new LinkedList<>();
        while (right < n) {
            int cur = arr[right++];
            if (!window.contains(cur)) {
                window.add(cur);
            } else {
                maxLen = Math.max(maxLen, window.size());
                while (window.contains(cur)) {
                    left++;
                    window.removeFirst();
                }
                window.add(cur);
            }
        }
        return maxLen;
    }

    public TreeNode mirrorTree(TreeNode root) {
        return buildTree(root);
    }

    public TreeNode buildTree(TreeNode root) {
        if (root == null) {
            return null;
        }
        TreeNode newRoot = new TreeNode(root.val);
        newRoot.left = buildTree(root.right);
        newRoot.right = buildTree(root.left);
        return newRoot;
    }

    public int[] exchange(int[] nums) {
        int n = nums.length;
        int left = 0, right = n - 1;
        while (left < right) {
            while (nums[left] % 2 != 0 && left < right) {
                left++;
            }
            while (nums[right] % 2 == 0 && left < right) {
                right--;
            }
            if (left < right) {
                swap(nums, left, right);
            }
        }
        return nums;
    }

    public void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    public boolean isStraight(int[] nums) {
        Arrays.sort(nums);
        int zeroNum = 0;
        int totalGaps = 0;
        for (int i = 0; i < 5; i++) {
            if (nums[i] == 0) {
                zeroNum++;
            } else {
                for (int j = i + 1; j < 5; j++) {
                    if (nums[j] == nums[j - 1]) {
                        return false;
                    }
                    if (nums[j] - nums[j - 1] != 1) {
                        totalGaps += nums[j] - (nums[j - 1] + 1);
                        i = j - 1;
                        break;
                    }
                }
            }
        }
        return zeroNum >= totalGaps;
    }

    public boolean isBalanced(TreeNode root) {
        if (root == null) return true;
        return depth(root) != -1;
    }

    private int depth(TreeNode root) {
        if (root == null) return 0;
        int left = depth(root.left);
        if (left == -1) {
            return -1;
        }
        int right = depth(root.right);
        if (right == -1) {
            return -1;
        }
        return Math.abs(left - right) < 2 ? Math.max(left, right) + 1 : -1;
    }

    public int lastRemaining(int n, int m) {
        if (n <= 1) {
            return 0;
        }
        int index = lastRemaining(n - 1, m);
        return (index + m) % n;
    }

    public boolean isHappy(int n) {
        HashSet<Integer> pows = new HashSet<>();
        while (true) {
            int sum = 0;
            while (n >= 10) {
                int curBit = n % 10;
                sum += curBit * curBit;
                n /= 10;
            }
            sum += n * n;
            if (sum == 1) {
                return true;
            }
            if (pows.contains(sum)) {
                return false;
            }
            pows.add(sum);
            n = sum;
        }
    }

    public int lengthOfLongestSubstring(String s) {
        int n = s.length();
        HashMap<Character, Integer> window = new HashMap<>();
        int left = 0, right = 0;
        int maxLen = 0;
        while (right < n) {
            char ch = s.charAt(right++);
            window.put(ch, window.getOrDefault(ch, 0) + 1);
            while (window.get(ch) > 1) {
                maxLen = Math.max(maxLen, right - left - 1);
                char c = s.charAt(left++);
                if (window.get(c) > 1) {
                    window.put(c, window.get(c) - 1);
                } else if (window.get(c) == 1) {
                    window.remove(c);
                }
            }
        }
        maxLen = Math.max(maxLen, right - left);
        return maxLen;
    }

    public String minNumber(int[] nums) {
        int n = nums.length;
        String[] strs = new String[n];
        for (int i = 0; i < n; i++) {
            strs[i] = String.valueOf(nums[i]);
        }
        Arrays.sort(strs, new Comparator<String>() {
            @Override
            public int compare(String o1, String o2) {
                return (o1 + o2).compareTo(o2 + o1);
            }
        });
        StringBuilder ans = new StringBuilder();
        for (int i = 0; i < n; i++) {
            ans.append(strs[i]);
        }
        return ans.toString();
    }

    public int[][] findContinuousSequence(int target) {
        LinkedList<Integer> window = new LinkedList<>();
        List<List<Integer>> ans = new LinkedList<>();
        int left = 1, right = 1;
        int sum = 0;
        while (right <= target / 2 + 1) {
            sum += right;
            window.add(right);
            right++;
            while (sum >= target) {
                if (sum == target) {
                    ans.add(new LinkedList<>(window));
                }
                sum -= window.getFirst();
                window.removeFirst();
                left++;
            }
        }
        int n = ans.size();
        int[][] res = new int[n][];
        for (int i = 0; i < n; i++) {
            res[i] = new int[ans.get(i).size()];
            for (int j = 0; j < ans.get(i).size(); j++) {
                res[i][j] = ans.get(i).get(j);
            }
        }
        return res;
    }

    public int cuttingRope(int n) {
        int[] dp = new int[n + 1];
        dp[1] = 1;
        dp[2] = 2;
        dp[3] = 3;
        dp[4] = 4;
        for (int i = 5; i <= n; i++) {
            for (int j = 1; j < i; j++) {
                dp[i] = Math.max(dp[i], (dp[i - j] % 1000000007) * (dp[j] % 1000000007));
            }
        }
        return dp[n];
    }

    class Node {
        int val;
        Node next;
        Node random;

        public Node(int val) {
            this.val = val;
            this.next = null;
            this.random = null;
        }
    }

    public boolean isSubStructure(TreeNode A, TreeNode B) {
        if (A == null || B == null) {
            return false;
        }
        String a = serialize(A, "");
        String b = serialize(B, "");
        int bLen = b.length();
        int aLen = a.length();
        if (bLen > aLen) {
            return false;
        }
        for (int i = 0; i < aLen - bLen; i++) {
            if (a.substring(i, i + bLen).equals(b)) {
                return true;
            }
        }
        return false;
    }

    public String serialize(TreeNode root, String ans) {
        if (root == null) {
            return "";
        }
        ans = ans + root.val;
        ans = ans + ',';
        ans += (serialize(root.left, ""));
        ans += (serialize(root.right, ""));
        ans = ans.substring(0, ans.length() - 1);
        return ans;
    }

    public String[] permutation(String s) {
        boolean[] visited = new boolean[s.length()];
        backtrack(s, visited);
        String[] ans = new String[per.size()];
        for (int i = 0; i < per.size(); i++) {
            ans[i] = per.get(i);
        }
        return ans;
    }

    List<String> per = new LinkedList<>();
    HashSet<String> chosen = new HashSet<>();
    String Trac = "";

    public void backtrack(String s, boolean[] visited) {
        if (Trac.length() == s.length() && !chosen.contains(Trac)) {
            per.add(Trac);
            chosen.add(Trac);
            return;
        }
        for (int i = 0; i < s.length(); i++) {
            if (visited[i]) {
                continue;
            }
            Trac = Trac + s.charAt(i);
            visited[i] = true;
            backtrack(s, visited);
            visited[i] = false;
            Trac = Trac.substring(0, Trac.length() - 1);
        }
    }

    public List<List<Integer>> levelOrder(TreeNode root) {
        if (root == null) {
            return new LinkedList<>();
        }
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        int level = 0;
        List<List<Integer>> ans = new LinkedList<>();
        while (!q.isEmpty()) {
            int sz = q.size();
            LinkedList<Integer> thisLeval = new LinkedList<>();
            for (int i = 0; i < sz; i++) {
                TreeNode cur = q.poll();
                if (level % 2 == 0) {
                    thisLeval.addLast(cur.val);
                } else {
                    thisLeval.addFirst(cur.val);
                }

                if (cur.left != null) {
                    q.offer(cur.left);
                }
                if (cur.right != null) {
                    q.offer(cur.right);
                }
            }
            ans.add(thisLeval);
            level++;
        }
        return ans;
    }

    public int findNthDigit(int n) {
        int num = 0;
        int curLen = 0;
        String curNum = "";
        int bitIndex = 0;
        while (true) {
            curNum = String.valueOf(++num);
            curLen += curNum.length();
            if (curLen >= n) {
                bitIndex = curNum.length() - (curLen - n) - 1;
                break;
            }
        }
        return curNum.charAt(bitIndex) - '0';
    }

    public boolean verifyPostorder(int[] postorder) {
        return check(postorder);
    }

    public boolean check(int[] postorder) {
        if (postorder.length == 0 || postorder.length == 1) {
            return true;
        }
        int n = postorder.length;
        int root = postorder[n - 1];
        int leftBound = 0, rightBound = 0;//leftbound指的是左子树的右边界，rightbound指的是右子树的左边界，左右子树均左闭右开
        for (int i = n - 2; i >= 0; i--) {
            if (postorder[i] < root) {
                leftBound = i + 1;
                rightBound = i + 1;
                break;
            }
        }
        for (int i = 0; i < leftBound; i++) {
            if (postorder[i] >= root) {
                return false;
            }
        }
        return check(Arrays.copyOfRange(postorder, 0, leftBound)) && check(Arrays.copyOfRange(postorder, rightBound, n - 1));
    }

    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        if (root == null) {
            return Ans;
        }
        branch.add(root.val);
        backtrack(root, sum, root.val);
        return Ans;
    }

    LinkedList<Integer> branch = new LinkedList<>();
    List<List<Integer>> Ans = new LinkedList<>();

    public void backtrack(TreeNode root, int sum, int curSum) {
        if (root.left == null && root.right == null) {
            if (curSum == sum) {
                Ans.add(new LinkedList<>(branch));
            }
            return;
        }
        if (root.left != null) {
            branch.add(root.left.val);
            backtrack(root.left, sum, curSum + root.left.val);
            branch.removeLast();
        }
        if (root.right != null) {
            branch.add(root.right.val);
            backtrack(root.right, sum, curSum + root.right.val);
            branch.removeLast();
        }
    }

    public int reversePairs(int[] nums) {
        this.nums = nums;
        this.temp = new int[nums.length];
        return mergeSort(0, nums.length - 1);
    }

    int[] nums;
    int[] temp;

    public int mergeSort(int left, int right) {
        if (left >= right) {
            return 0;
        }
        int mid = (left + right) / 2;
        int res = mergeSort(left, mid) + mergeSort(mid + 1, right);
        int i = left, j = mid + 1;
        for (int k = left; k <= right; k++) {
            temp[k] = nums[k];
        }
        for (int k = left; k <= right; k++) {
            if (i == mid + 1) {
                nums[k] = temp[j++];
            } else if (j == right + 1) {
                nums[k] = temp[i++];
            } else if (temp[i] <= temp[j]) {
                nums[k] = temp[i++];
            } else if (temp[i] > temp[j]) {
                nums[k] = temp[j++];
                res += mid + 1 - i;
            }
        }
        return res;
    }

    public int[] singleNumbers(int[] nums) {
        int xy = 0;
        for (int n : nums) {
            xy ^= n;
        }
        int m = 1;
        while ((xy & m) == 0) {
            m <<= 1;
        }
        int x = 0, y = 0;
        for (int n : nums) {
            if ((n & m) == 0) {
                x ^= n;
            } else {
                y ^= n;
            }
        }
        return new int[]{x, y};
    }

    public int[] constructArr(int[] a) {
        int n = a.length;
        if (n == 0 || n == 1) {
            return new int[]{};
        }
        int[] leftMul = new int[n];
        int[] rightMul = new int[n];
        leftMul[0] = rightMul[n - 1] = 1;
        for (int i = 1; i < n; i++) {
            leftMul[i] = leftMul[i - 1] * a[i - 1];
            rightMul[n - 1 - i] = rightMul[n - i] * a[n - i];
        }
        int[] ans = new int[n];
        for (int i = 0; i < n; i++) {
            ans[i] = leftMul[i] * rightMul[i];
        }
        return ans;
    }



    public ListNode solve (ListNode[] a) {
        // write code here
        int n=a.length;
        ListNode[]p=new ListNode[n];
        int nullptr=0;
        for(int i=0;i<n;i++){
            if(a[i]!=null){
                p[i]=a[i];
            }else {
                p[i]=null;
                nullptr++;
            }
        }
        ListNode head=new ListNode(-1);
        ListNode cur=head;
        while(true) {
            for (int i = 0; i < n; i++) {
                if (p[i] != null) {
                    cur.next = new ListNode(p[i].val);
                    cur = cur.next;
                    p[i] = p[i].next;
                    if(p[i]==null){
                        nullptr++;
                        if(nullptr>=n){
                            break;
                        }
                    }
                }
            }
            if(nullptr>=n){
                break;
            }
        }
        return head.next;
    }
    public int totalNQueens(int n) {
        char[][]board=new char[n][n];
        for(char[]b:board){
            Arrays.fill(b,'.');
        }
        backtrack(board,n,0);
        return res;
    }
    int res=0;
    void backtrack(char[][]board,int n,int row){
        if(row==n){
            res++;
            return;
        }
        for(int i=0;i<n;i++){
            if(isValid(board,row,i)){
                board[row][i]='Q';
                backtrack(board,n,row+1);
                board[row][i]='.';
            }
        }
    }
    boolean isValid(char[][]board,int row,int col){
        int m=board.length;
        int n=board[0].length;
        for(int i=0;i<n;i++){
            if(board[row][i]=='Q'){
                return false;
            }
        }
        for(int i=0;i<m;i++){
            if(board[i][col]=='Q'){
                return false;
            }
        }
        for(int i=row-1,j=col-1;i>=0&&j>=0;i--,j--){
            if(board[i][j]=='Q'){
                return false;
            }
        }
        for(int i=row-1,j=col+1;i>=0&&j<n;i--,j++){
            if(board[i][j]=='Q'){
                return false;
            }
        }
        for(int i=row+1,j=col-1;i<m&&j>=0;i++,j--){
            if(board[i][j]=='Q'){
                return false;
            }
        }
        for(int i=row+1,j=col+1;i<m&&j<n;i++,j++){
            if(board[i][j]=='Q'){
                return false;
            }
        }
        return true;
    }
    public static void main(String[] args) {
        workout wo = new workout();
        int[]a1={1,3},a2={7},a3={2};
        ListNode h1=new ListNode(a1),h2=new ListNode(a2),h3=new ListNode(a3);
        h2=null;
        ListNode[]arr={h1,h2,h3};
        System.out.println(wo.solve(arr));
        Scanner in = new Scanner(System.in);
        System.out.println(wo.totalNQueens(4));
    }
}
