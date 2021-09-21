import sun.applet.AppletResourceLoader;

import java.util.*;
import java.lang.*;

public class Solution {
    HashMap<Integer, Integer> numMap = new HashMap<>();


//    public TreeNode buildTree(int[] preorder, int[] inorder) {
//        for (int i = 0; i < inorder.length; i++) {
//            mp.put(inorder[i], i);
//        }
//        return rec(0, preorder, 0, preorder.length - 1);
//    }

    public TreeNode rec(int root, int[] preorder, int left, int right) {
        if (left > right) {
            return null;
        }
        TreeNode rootNode = new TreeNode(preorder[root]);
        int i = numMap.get(preorder[root]);
        rootNode.left = rec(root + 1, preorder, left, i - 1);
        rootNode.right = rec(root + i - left + 1, preorder, i + 1, right);
        return rootNode;
    }


    public void quick_sort(int[] arr, int low, int high) {
        int p = low, i = low, j = high;
        if (low >= high) {
            return;
        }
        while (i < j) {
            while (arr[j] >= arr[p] && i < j) {
                j--;
            }
            while (arr[i] <= arr[p] && i < j) {
                i++;
            }
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
        int temp = arr[i];
        arr[i] = arr[p];
        arr[p] = temp;
        quick_sort(arr, low, i - 1);
        quick_sort(arr, i + 1, high);
    }

    public int minArray(int[] numbers) {
        int start = 0, end = numbers.length - 1;
        if (numbers[end] > numbers[start]) {
            return numbers[start];
        }
        int middle = 0;
        while (start <= end) {
            middle = (start + end) / 2;
            if (end - start <= 1) {
                middle = end;
                break;
            }
            if (numbers[middle] > numbers[end]) {
                start = middle;
            } else if (numbers[middle] < numbers[end]) {
                end = middle;
            } else {
                end--;
            }
        }
        return numbers[middle];
    }

    public boolean exist(char[][] board, String word) {
        int m = board.length, n = board[0].length;
        boolean[][] visited = new boolean[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                boolean ans = recc(board, visited, i, j, 0, word);
                if (ans) {
                    return true;
                } else {
                    for (boolean[] arr : visited) {
                        Arrays.fill(arr, false);
                    }
                }
            }
        }
        return false;
    }

    //    public int visited_num=0;
    public boolean recc(char[][] board, boolean[][] visited, int x, int y, int index, String word) {
        if (index == word.length() - 1 && board[x][y] == word.charAt(index)) {
            return true;
        }
        if (board[x][y] != word.charAt(index)) {
            return false;
        }
        int m = board.length, n = board[0].length;
        int[] dx = {1, 0, -1, 0};
        int[] dy = {0, 1, 0, -1};
        visited[x][y] = true;
        boolean result = false;
        for (int i = 0; i < 4; i++) {
            int nx = x + dx[i], ny = y + dy[i];
            if (nx >= 0 && nx < m && ny >= 0 && ny < n && !visited[nx][ny]) {
                result = recc(board, visited, nx, ny, index + 1, word);
                if (result) {
                    return true;
                }
            }
        }
        visited[x][y] = false;
        return result;
    }

    public int movingCount(int m, int n, int k) {
        if (m <= 0 || n <= 0) {
            return 0;
        }
        boolean[][] visited = new boolean[m][n];
        int ans = dfs(0, 0, m, n, k, visited);
        return ans;
    }

    public int dfs(int x, int y, int m, int n, int k, boolean[][] visited) {
        int ans = 0;
        int[] dx = {1, 0, -1, 0};
        int[] dy = {0, 1, 0, -1};
        visited[x][y] = true;
        ans += 1;
        for (int i = 0; i < 4; i++) {
            int nx = x + dx[i], ny = y + dy[i];
            if (nx >= 0 && nx < m && ny >= 0 && ny < n && !visited[nx][ny] && sumDigits(nx) + sumDigits(ny) <= k) {
                //     visited[nx][ny] = true;

                ans += dfs(nx, ny, m, n, k, visited);
            }
        }
        return ans;
    }

    public int hammingWeight(int n) {
        int ans = 0;
        int flag = 1;
        while (n != 0) {
            ans++;
            n = n & (n - 1);
        }
        return ans;
    }


    public int sumDigits(int n) {
        int ans = 0;
        while (n >= 10) {
            ans += n % 10;
            n /= 10;
        }
        ans += n;
        return ans;
    }

    public double myPow(double x, int n) {
        if (n == 0) {
            return 1;
        }
        if (n == 1) {
            return x;
        }
        if (x == 0) {
            return 0;
        }
        int b = n;
        double a = x;

        double result = 0.0;
        if (b > 0) {
            result = myPow(a, b >> 1);
        } else {
            if (b >> 1 == -1) {
                return result * result * a;
            }
            result = myPow(a, b >> 1);
        }
        result *= result;

        if ((b & 1) != 0) {
            result *= a;
        }
        return result;
    }

    public ListNode deleteNode(ListNode head, int val) {
        if (head == null) {
            return head;
        }
        if (head.val == val) {
            return head.next;
        }
        ListNode pre = head, cur = pre.next;
        while (cur != null && cur.val != val) {
            pre = cur;
            cur = cur.next;
        }
        if (cur.next != null) {
            pre.next = cur.next;
        } else {
            pre.next = null;
        }
        return head;
    }

    public ListNode deleteDuplicates(ListNode head) {
        if (head == null) {
            return null;
        }
        ListNode pre = head, cur = pre.next;
        while (pre != null && cur != null) {
            if (pre.val == cur.val) {
                pre.next = cur.next;
                cur = pre.next;
            } else {
                pre = cur;
                cur = cur.next;
            }
        }
        return head;
    }


//    List<List<Integer>> res = new LinkedList<>();
//
//    public List<List<Integer>> permute(int[] nums) {
//        LinkedList<Integer> track = new LinkedList<>();
//        backtrack(nums, track);
//        return res;
//    }
//
//    void backtrack(int[] nums, LinkedList<Integer> track) {
//        // 触发结束条件
//        if (track.size() == nums.length) {
//            res.add(new LinkedList<>(track));
//            return;
//        }
//        for (int i = 0; i < nums.length; i++) {
//            if (track.contains(nums[i])) {
//                continue;
//            }
//            track.add(nums[i]);
//            backtrack(nums, track);
//            track.removeLast();
//        }
//    }

    List<List<String>> ans = new LinkedList<>();

    public List<List<String>> solveNQueens(int n) {
        char[][] board = new char[n][n];
        for (int i = 0; i < n; i++) {
            Arrays.fill(board[i], '.');
        }
        myTrack(board, 0);
        return ans;
    }

    void myTrack(char[][] board, int row) {
        if (row == board.length) {
            ans.add(arr2list(board));
            return;
        }
        for (int i = 0; i < board[0].length; i++) {
            if (isValid(board, row, i)) {
//                StringBuilder s=board.get(row);
//                s.setCharAt(i,'Q');
                board[row][i] = 'Q';
                myTrack(board, row + 1);
                board[row][i] = '.';
            }
        }
    }

    public List<String> arr2list(char[][] board) {
        List<String> ans = new LinkedList<>();
        for (char[] b : board) {
            ans.add(new String(b));
        }
        return ans;
    }

    public boolean isValid(char[][] board, int row, int col) {
        int m = board.length;
        int n = board[0].length;
        for (int i = 0; i < m; i++) {
            if (board[i][col] == 'Q') {
                return false;
            }
        }
        for (int i = 0; i < n; i++) {
            if (board[row][i] == 'Q') {
                return false;
            }
        }
        for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {//该点左上
            if (board[i][j] == 'Q') {
                return false;
            }
        }
        for (int i = row + 1, j = col + 1; i < m && j < n; i++, j++) {//该点右下
            if (board[i][j] == 'Q') {
                return false;
            }
        }
        for (int i = row + 1, j = col - 1; i < m && j >= 0; i++, j--) {//该点左下
            if (board[i][j] == 'Q') {
                return false;
            }
        }
        for (int i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++) {//该点右上
            if (board[i][j] == 'Q') {
                return false;
            }
        }
        return true;
    }

    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        Queue<TreeNode> q = new LinkedList<>();
        int depth = 1;
        q.offer(root);
        while (!q.isEmpty()) {
            int sz = q.size();
            for (int i = 0; i < sz; i++) {
                TreeNode cur = q.poll();
                if (cur.left == null && cur.right == null) {
                    return depth;
                }
                if (cur.left != null) {
                    q.offer(cur.left);
                }
                if (cur.right != null) {
                    q.offer(cur.right);
                }
            }
            depth++;
        }
        return -1;
    }

    public int openLock(String[] deadends, String target) {
        String start = "0000";
        List<String> de = Arrays.asList(deadends);
        Set<String> visited = new HashSet<>();
        if (de.contains(start)) {
            return -1;
        }
        Queue<String> q = new LinkedList<>();
        q.offer(start);
        int steps = 0;
        while (!q.isEmpty()) {
            int sz = q.size();
            for (int i = 0; i < sz; i++) {
                String begin = q.poll();
                if (begin.equals(target)) {
                    return steps;
                }
                if (de.contains(begin)) {
                    continue;
                }
                for (int j = 0; j < 4; j++) {
                    String nbegin = plusBit(begin, j);
                    if (!visited.contains(nbegin)) {
                        q.offer(nbegin);
                        visited.add(nbegin);
                    }
                }
                for (int j = 0; j < 4; j++) {
                    String nbegin = minusBit(begin, j);
                    if (!visited.contains(nbegin)) {
                        q.offer(nbegin);
                        visited.add(nbegin);
                    }
                }
            }
            steps++;
        }
        return -1;
    }

    String plusBit(String s, int index) {
        char[] c = s.toCharArray();
        if (c[index] == '9') {
            c[index] = '0';
        } else {
            c[index] += 1;
        }
        return new String(c);
    }

    String minusBit(String s, int index) {
        char[] c = s.toCharArray();
        if (c[index] == '0') {
            c[index] = '9';
        } else {
            c[index] -= 1;
        }
        return new String(c);
    }

    public int search(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else if (nums[mid] > target) {
                right = mid - 1;
            }
        }
        return -1;
    }

    public int[] searchRange(int[] nums, int target) {
        int[] ans = {-1, -1};
        int left = 0, right = nums.length - 1;
        while (left <= right) {//这里要寻找左右边界。我这里准备寻找左边界
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                right = mid - 1;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else if (nums[mid] > target) {
                right = mid - 1;
            }
        }
        if (left >= nums.length || nums[left] != target) {
            return new int[]{-1, -1};
        }
        ans[0] = ans[1] = left;
        for (int i = left; i < nums.length; i++) {
            if (i == nums.length - 1 && nums[i] == target) {
                ans[1] = i;
                break;
            }
            if (nums[i] != target) {
                right = i - 1;
                ans[1] = right;
                break;
            }
        }
        return ans;
    }

    public String minWindow(String s, String t) {
        int MAX_LEN = 10000010;
        HashMap<Character, Integer> need = new HashMap<>();
        HashMap<Character, Integer> window = new HashMap<>();
        for (int i = 0; i < t.length(); i++) {
            need.put(t.charAt(i), need.getOrDefault(t.charAt(i), 0) + 1);
        }
        int left = 0, right = 0;
        int valid = 0;
        int start = 0, len = MAX_LEN;
        while (right < s.length()) {
            char d = s.charAt(right++);
            if (need.containsKey(d)) {
                window.put(d, window.getOrDefault(d, 0) + 1);
                if (window.get(d).equals(need.get(d))) {
                    valid++;
                }
            }
            while (valid == need.size()) {
                if (right - left < len) {
                    len = right - left;
                    start = left;
                }
                char c = s.charAt(left++);
                if (need.containsKey(c)) {
                    if (window.get(c).equals(need.get(c))) {
                        valid--;
                    }
                    window.put(c, window.get(c) - 1);
                }
            }
        }
        return len == MAX_LEN ? "" : s.substring(start, right);
    }

    public boolean checkInclusion(String s1, String s2) {
        HashMap<Character, Integer> need = new HashMap<>();
        HashMap<Character, Integer> window = new HashMap<>();
        for (int i = 0; i < s1.length(); i++) {
            need.put(s1.charAt(i), need.getOrDefault(s1.charAt(i), 0) + 1);
        }
        int left = 0, right = 0;
        int valid = 0;
//        int start=0,len=MAX_LEN;
        while (right < s2.length()) {
            char d = s2.charAt(right++);
            if (need.containsKey(d)) {
                window.put(d, window.getOrDefault(d, 0) + 1);
                if (window.get(d).equals(need.get(d))) {
                    valid++;
                }
            }
            while (right - left >= s1.length()) {
                if (valid == need.size() && right - left == s1.length()) {
                    return true;
                }
                char c = s2.charAt(left++);
                if (need.containsKey(c)) {
                    if (window.get(c).equals(need.get(c))) {
                        valid--;
                    }
                    window.put(c, window.get(c) - 1);
                }
            }
        }
        return false;
    }

    public List<Integer> findAnagrams(String s, String p) {
        HashMap<Character, Integer> need = new HashMap<>();
        HashMap<Character, Integer> window = new HashMap<>();
        List<Integer> ans = new LinkedList<>();
        for (int i = 0; i < p.length(); i++) {
            need.put(p.charAt(i), need.getOrDefault(p.charAt(i), 0) + 1);
        }
        int left = 0, right = 0;
        int valid = 0;
        while (right < s.length()) {
            char d = s.charAt(right++);
            if (need.containsKey(d)) {
                window.put(d, window.getOrDefault(d, 0) + 1);
                if (window.get(d).equals(need.get(d))) {
                    valid++;
                    if (valid == need.size()) {
                        ans.add(left);
                    }
                }
            }
            while (right - left >= p.length()) {
                char c = s.charAt(left++);

                if (need.containsKey(c)) {
                    if (window.get(c).equals(need.get(c))) {
                        valid--;
                    }
                    window.put(c, window.get(c) - 1);
                }
            }
        }
        return ans;
    }

    public int lengthOfLongestSubstring(String s) {
        HashMap<Character, Integer> window = new HashMap<>();
        List<Integer> record = new LinkedList<>();
        int left = 0, right = 0, len = 0;
        while (right < s.length()) {
            char d = s.charAt(right++);
            if (!window.containsKey(d)) {
                window.put(d, 1);
                len++;
            } else {
                record.add(len);
                if (right < s.length() && s.charAt(right) != s.charAt(left)) {
                    len--;
                }
                left++;
            }
        }
        int maxNum = -1;
        for (int i : record) {
            maxNum = Math.max(maxNum, i);
        }
        return maxNum;
    }

    public int maxProfit(int[] prices) {//在系统里面每道题的函数名称都一样
        int n = prices.length;
        int max_k = 2;
        int[][][] dp = new int[n][max_k + 1][2];
        for (int i = 0; i < n; i++) {
            if (i - 1 == -1) {
                dp[i][0][0] = 0;
                dp[i][0][1] = Integer.MIN_VALUE;
                dp[i][1][0] = 0;
                dp[i][1][1] = -prices[i];
                dp[i][2][0] = Integer.MIN_VALUE;
                dp[i][2][1] = Integer.MIN_VALUE;
                continue;
            }
            for (int k = max_k; k > 0; k--) {
                dp[i][k][0] = Math.max(dp[i - 1][k][0], dp[i - 1][k][1] + prices[i]);
                dp[i][k][1] = Math.max(dp[i - 1][k][1], dp[i - 1][k - 1][0] - prices[i]);
            }
        }
        return Math.max(dp[n - 1][max_k][0], dp[n - 1][1][0]);
    }

    public ListNode reverseBetween(ListNode head, int m, int n) {
        if (m == 1) {
            return reverseN(head, n);
        }
        head.next = reverseBetween(head.next, m - 1, n - 1);
        return head;
    }

    ListNode succecor = null;

    public ListNode reverseN(ListNode head, int n) {
        if (n == 1) {
            succecor = head.next;
            return head;
        }
        ListNode last = reverseN(head.next, n - 1);
        head.next.next = head;
        head.next = null;
        return last;
    }

    public ListNode reverse(ListNode a, ListNode b) {
        ListNode pre = null, cur = a, nxt;
        while (cur != b) {
            nxt = cur.next;
            cur.next = pre;
            pre = cur;
            cur = nxt;
        }
        return pre;
    }

    public ListNode reverseKGroup(ListNode head, int k) {
        if (head == null) {
            return null;
        }
        ListNode a = head, b = head;
        for (int i = 0; i < k; i++) {
            if (b == null) {
                return head;
            }
            b = b.next;
        }
        ListNode newHead = reverse(a, b);
        a.next = reverseKGroup(b, k);
        return newHead;
    }

    public TreeNode constructMaximumBinaryTree(int[] nums) {
        if (nums.length == 0) {
            return null;
        }
        int max_value = Integer.MIN_VALUE;
        int pivot = -1;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] > max_value) {
                max_value = nums[i];
                pivot = i;
            }
        }
        TreeNode root = new TreeNode(nums[pivot]);
        int[] leftTreeNums = Arrays.copyOfRange(nums, 0, pivot);
        int[] rightTreeNums = Arrays.copyOfRange(nums, pivot + 1, nums.length);
        root.left = constructMaximumBinaryTree(leftTreeNums);
        root.right = constructMaximumBinaryTree(rightTreeNums);
        return root;
    }

    public TreeNode buildTree(int[] inorder, int[] postorder) {
        int n = inorder.length;
        if (inorder.length == 0 || postorder.length == 0) {
            return null;
        }
        TreeNode root = new TreeNode(postorder[n - 1]);
        int pivot = -1;
        for (int i = 0; i < inorder.length; i++) {
            if (inorder[i] == postorder[n - 1]) {
                pivot = i;
                break;
            }
        }
        int right_len = inorder.length - pivot - 1;
        int left_len = pivot;
        root.left = buildTree(Arrays.copyOfRange(inorder, 0, pivot),
                Arrays.copyOfRange(postorder, 0, left_len));
        root.right = buildTree(Arrays.copyOfRange(inorder, pivot + 1, n),
                Arrays.copyOfRange(postorder, n - right_len - 1, n - 1));
        return root;
    }

    public List<TreeNode> findDuplicateSubtrees(TreeNode root) {
        traverse(root);
        return answer;
    }

    List<TreeNode> answer = new LinkedList<>();
    HashMap<String, Integer> memo = new HashMap<>();

    public String traverse(TreeNode root) {
        if (root == null) {
            return "#";
        }
        String left = traverse(root.left);
        String right = traverse(root.right);
        String subtree = left + "," + right + "," + root.val;
        int freq = memo.getOrDefault(subtree, 0);
        if (freq == 1) {
            answer.add(root);
        }
        memo.put(subtree, freq + 1);
        return subtree;
    }

    int sum = 0;

    public TreeNode bstToGst(TreeNode root) {
        Traverse(root);
        return root;
    }

    public void Traverse(TreeNode root) {
        if (root == null) {
            return;
        }
        Traverse(root.right);
        sum += root.val;
        root.val = sum;
        Traverse(root.left);
    }

    public TreeNode deleteNode(TreeNode root, int key) {
        if (root == null) {
            return null;
        }
        if (root.val == key) {
            if (root.left == null && root.right == null) {
                return null;
            }
            if (root.right == null) {
                return root.left;
            }
            if (root.left == null) {
                return root.right;
            }
            if (root.left != null && root.right != null) {
                TreeNode maxNode = getMax(root.left);
                root.val = maxNode.val;
                root.left = deleteNode(root.left, maxNode.val);
            }
        } else if (root.val < key) {
            root.right = deleteNode(root.right, key);
        } else if (root.val > key) {
            root.left = deleteNode(root.left, key);
        }
        return root;
    }

    public TreeNode getMax(TreeNode node) {
        while (node.right != null) {
            node = node.right;
        }
        return node;
    }

    public TreeNode insertIntoBST(TreeNode root, int val) {
        if (root == null) {
            return new TreeNode(val);
        }
        if (val < root.val) {
            root.left = insertIntoBST(root.left, val);
        } else if (val > root.val) {
            root.right = insertIntoBST(root.right, val);
        }
        return root;
    }

    int mysum = 0;
    int myAns = 0;

    public int findTargetSumWays(int[] nums, int S) {
        dfs(nums, 0, S);
        return myAns;
    }

    public void dfs(int[] nums, int i, int S) {
        if (mysum == S && i == nums.length) {
            myAns++;
        }
        if (i == nums.length) {
            return;
        }
        mysum += nums[i];
        dfs(nums, i + 1, S);
        mysum -= nums[i];
        mysum -= nums[i];
        dfs(nums, i + 1, S);
        mysum += nums[i];
    }

    public int lengthOfLIS(int[] nums) {
        int n = nums.length;
        int[] dp = new int[n];
        Arrays.fill(dp, 1);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
        }
        int ans = 0;
        for (int i = 0; i < n; i++) {
            ans = Math.max(ans, dp[i]);
        }
        return ans;
    }

    public int maxEnvelopes(int[][] envelopes) {
        int n = envelopes.length;
        Arrays.sort(envelopes, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] == o2[0] ? o2[1] - o1[1] : o1[0] - o2[0];
            }
        });
        int[] height = new int[n];
        for (int i = 0; i < n; i++) {
            height[i] = envelopes[i][1];
        }
        return lengthOfLIS(height);
    }

    public int maxSubArray(int[] nums) {
        int n = nums.length;
        int dp = nums[0];
        int Max = Integer.MIN_VALUE;
        for (int i = 1; i < n; i++) {
            dp = Math.max(nums[i], dp + nums[i]);
            Max = Math.max(Max, dp);
        }
        return Max;
    }

    public int longestCommonSubsequence(String text1, String text2) {
        int m = text1.length(), n = text2.length();
        int[][] dp = new int[m + 1][n + 1];
        Arrays.fill(dp[0], 0);
        for (int i = 0; i <= m; i++) {
            dp[i][0] = 0;
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[m][n];
    }

    public int minDistance(String word1, String word2) {
        int m = word1.length(), n = word2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 0; i <= m; i++) {
            dp[i][0] = i;
        }
        for (int i = 0; i <= n; i++) {
            dp[0][i] = i;
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1]) + 1;
                }
            }
        }
        return dp[m][n];
    }

    public int Min(int a, int b, int c) {
        return Math.min(a, Math.min(b, c));
    }

    public int longestPalindromeSubseq(String s) {
        int n = s.length();
        int[][] dp = new int[n][n];
        for (int i = 0; i < n; i++) {
            dp[i][i] = 1;
        }
        for (int i = n - 2; i >= 0; i--) {
            for (int j = i + 1; j < n; j++) {
                if (s.charAt(i) == s.charAt(j)) {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                } else {
                    dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[0][n - 1];
    }

    public int minInsertions(String s) {
        int n = s.length();
        int[][] dp = new int[n][n];
        for (int i = n - 2; i >= 0; i--) {
            for (int j = i + 1; j < n; j++) {
                if (s.charAt(i) == s.charAt(j)) {
                    dp[i][j] = dp[i + 1][j - 1];
                } else {
                    dp[i][j] = Math.min(dp[i + 1][j], dp[i][j - 1]) + 1;
                }
            }
        }
        return dp[0][n - 1];
    }

    HashMap<String, Boolean> mem = new HashMap<>();

    public boolean isMatch(String s, String p) {
        return dp(s, 0, p, 0);
    }

    public boolean dp(String s, int i, String p, int j) {
        if (j == p.length()) {
            return i == s.length();
        }
        if (i == s.length()) {
            return isNullMatched(p.substring(j));
        }
        String key = i + "," + j;
        if (mem.containsKey(key)) {
            return mem.get(key);
        }
        boolean record = false;
        if (s.charAt(i) == p.charAt(j) || p.charAt(j) == '.') {
            if (j + 1 < p.length() && p.charAt(j + 1) == '*') {
                record = dp(s, i + 1, p, j) || dp(s, i + 1, p, j + 2) || dp(s, i, p, j + 2);
            } else {
                record = dp(s, i + 1, p, j + 1);
            }
        } else {
            if (j + 1 < p.length() && p.charAt(j + 1) == '*') {
                record = dp(s, i, p, j + 2);
            } else {
                record = false;
            }
        }
        mem.put(key, record);
        return record;
    }

    public boolean isNullMatched(String s) {
        if (s.length() % 2 != 0) {
            return false;
        }
        for (int i = 1; i < s.length(); i += 2) {
            if (s.charAt(i) != '*') {
                return false;
            }
        }
        return true;
    }

    public int rob(int[] nums) {
        int n = nums.length;
        int[] dp = new int[n];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);
        for (int i = 2; i < n; i++) {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);
        }
        return dp[n - 1];
    }

    public int rob2(int[] nums) {
        int n = nums.length;
        if (n == 0) {
            return 0;
        }
        if (n == 1) {
            return nums[0];
        }
        int[] dp = new int[n];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);
        for (int i = 2; i < n; i++) {
            if (i != n - 1) {
                dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);
            } else {
                dp[i] = dp[i - 1];
            }
        }
        int max1 = dp[n - 1];
        Arrays.fill(dp, 0);
        dp[n - 1] = nums[n - 1];
        dp[n - 2] = Math.max(nums[n - 1], nums[n - 2]);
        for (int i = n - 3; i >= 0; i--) {
            if (i != 0) {
                dp[i] = Math.max(dp[i + 1], dp[i + 2] + nums[i]);
            } else {
                dp[i] = dp[i + 1];
            }
        }
        int max2 = dp[0];
        return Math.max(max1, max2);
    }

    public int rob3(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int robThisNode = root.val + (root.left == null ? 0 : +rob3(root.left.left) + rob3(root.left.right))
                + (root.right == null ? 0 : +rob3(root.right.left) + rob3(root.right.right));
        int notRobThisNode = rob3(root.left) + rob3(root.right);
        int ans = Math.max(robThisNode, notRobThisNode);
        return ans;
    }

    public boolean isSymmetric(TreeNode root) {
        return root == null || check(root.left, root.right);
    }

    public boolean check(TreeNode p, TreeNode q) {
        if (p == null && q == null) {
            return true;
        }
        if ((p == null && q != null) || (p != null && q == null) || p.val != q.val) {
            return false;
        }
        return check(p.left, q.right) && check(p.right, q.left);
    }

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) {
            return null;
        }
        if (root == p || root == q) {
            return root;
        }
        if (root.val < p.val && root.val < q.val) {
            return lowestCommonAncestor(root.right, p, q);
        }
        if (root.val > p.val && root.val > q.val) {
            return lowestCommonAncestor(root.left, p, q);
        }
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if (left != null && right != null) {
            return root;
        }
        if (left == null && right == null) {
            return null;
        }
        return left == null ? right : left;
    }


    List<List<Integer>> myres = new LinkedList<>();
    LinkedList<Integer> track = new LinkedList<>();

    public List<List<Integer>> subsets(int[] nums) {
        backtrack(nums, 0);
        return myres;
    }

    public void backtrack(int[] nums, int start) {
        myres.add(new ArrayList<>(track));
        for (int i = start; i < nums.length; i++) {
            track.add(nums[i]);
            backtrack(nums, i + 1);
            track.removeLast();
        }
    }

    LinkedList<List<Integer>> coms = new LinkedList<>();
    LinkedList<Integer> mytrack = new LinkedList<>();

    public List<List<Integer>> combine(int n, int k) {
        backtrack(n, k, 1);
        return coms;
    }

    public void backtrack(int n, int k, int start) {
        if (mytrack.size() >= k) {
            coms.add(new LinkedList<>(mytrack));
            return;
        }
        for (int i = start; i <= n; i++) {
            mytrack.add(i);
            backtrack(n, k, i + 1);
            mytrack.removeLast();
        }
    }

    LinkedList<List<Integer>> per = new LinkedList<>();
    LinkedList<Integer> tracks = new LinkedList<>();

    public List<List<Integer>> permute(int[] nums) {
        backtrack(nums);
        return per;
    }

    public void backtrack(int[] nums) {
        if (tracks.size() == nums.length) {
            per.add(new LinkedList<>(tracks));
            return;
        }
        for (int i : nums) {
            if (tracks.contains(i)) {
                continue;
            }
            tracks.add(i);
            backtrack(nums);
            tracks.removeLast();
        }
    }

    public void solveSudoku(char[][] board) {
        backtrack(board, 0, 0);
    }

    public boolean backtrack(char[][] board, int row, int col) {
        if (row == 9) {
            return true;
        }
        if (col == 9) {
            return backtrack(board, row + 1, 0);
        }
        if (board[row][col] != '.') {
            return backtrack(board, row, col + 1);
        }

        for (char k = '1'; k <= '9'; k++) {
            if (!isValid(board, k, row, col)) {
                continue;
            }
            board[row][col] = k;
            if (backtrack(board, row, col + 1)) {
                return true;
            }
            board[row][col] = '.';

        }
        return false;
    }

    public boolean isValidSudoku(char[][] board) {
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (board[i][j] == '.') {
                    continue;
                }
                if (!isValid(board, board[i][j], i, j)) {
                    return false;
                }
            }
        }
        return true;
    }

    public boolean isValid(char[][] board, char target, int x, int y) {
        if (x < 0 || x >= 9 || y < 0 || y >= 9) {
            return false;
        }
        for (int i = 0; i < 9; i++) {//判断这一行是否可行
            if (board[x][i] == target) {
                return false;
            }
        }
        for (int i = 0; i < 9; i++) {//判断这一列是否可行
            if (board[i][y] == target) {
                return false;
            }
        }
        for (int i = (x / 3) * 3; i < (x / 3) * 3 + 3; i++) {
            for (int j = (y / 3) * 3; j < (y / 3) * 3 + 3; j++) {
                if (board[i][j] == target) {
                    return false;
                }
            }
        }
        return true;
    }

    public boolean isValid(String s) {
        HashMap<Character, Character> mp = new HashMap<>();
        mp.put(')', '(');
        mp.put(']', '[');
        mp.put('}', '{');
        Stack<Character> st = new Stack<>();
        char[] arr = s.toCharArray();
        LinkedList<Character> arrlist = new LinkedList<>();
        for (char ch : arr) {
            arrlist.add(ch);
        }
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == '(' || arr[i] == '[' || arr[i] == '{') {
                st.push(arr[i]);
                arrlist.removeFirst();
            } else {
                if (!st.isEmpty() && st.peek() == mp.get(arr[i])) {
                    st.pop();
                    arrlist.removeFirst();
                } else {
                    st.push(arr[i]);
                    arrlist.removeFirst();
                }
            }
        }
        return st.isEmpty();
    }

    LinkedList<String> strans = new LinkedList<>();

    public List<String> generateParenthesis(int n) {
        LinkedList<Character> track = new LinkedList<>();
        backtrack(2 * n, track);
        return strans;
    }

    public void backtrack(int n, LinkedList<Character> track) {
        if (n == 0) {
            if (isValid(list2str(track))) {
                strans.add((list2str(track)));
            }
            return;
        }
        track.add('(');
        backtrack(n - 1, track);
        track.removeLast();
        track.add(')');
        backtrack(n - 1, track);
        track.removeLast();
    }

    public String list2str(LinkedList<Character> track) {
        String ans = "";
        for (Character ch : track) {
            ans += ch;
        }
        return ans;
    }

    public int slidingPuzzle(int[][] board) {
        int[][] neibour = new int[][]{
                {1, 3},
                {0, 2, 4},
                {1, 5},
                {0, 4},
                {1, 5, 3},
                {2, 4}
        };
        HashSet<String> memo = new HashSet<>();
        String start = "";
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                start += board[i][j];
            }
        }
        String target = "123450";
        Queue<String> q = new LinkedList<>();
        q.offer(start);
        memo.add(start);
        int step = 0;
        while (!q.isEmpty()) {
            int sz = q.size();
            for (int i = 0; i < sz; i++) {
                String cur = q.poll();
                if (cur.equals(target)) {
                    return step;
                }
                int zero_index = 0;
                while (cur.charAt(zero_index) != '0') {
                    zero_index++;
                }
                for (int adj : neibour[zero_index]) {
                    String candidate = swap(cur, zero_index, adj);
                    if (!memo.contains(candidate)) {
                        q.offer(candidate);
                        memo.add(candidate);
                    }
                }
            }
            step++;
        }
        return -1;
    }

    public String swap(String s, int x, int y) {
        char[] arr = s.toCharArray();
        char temp = arr[x];
        arr[x] = arr[y];
        arr[y] = temp;
        return new String(arr);
    }

    public int[] twoSum(int[] nums, int target) {
        HashMap<Integer, Integer> mp = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            mp.put(nums[i], i);
        }
        for (int i = 0; i < nums.length; i++) {
            int other = target - nums[i];
            if (mp.containsKey(other)) {
                return new int[]{i, mp.get(other)};
            }
        }
        return new int[]{-1, -1};
    }

    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> ans = new LinkedList<>();
        Queue<TreeNode> q = new LinkedList<>();
        if (root == null) {
            return ans;
        }
        q.offer(root);
        while (!q.isEmpty()) {
            int sz = q.size();
            List<Integer> cur = new LinkedList<>();
            for (int i = 0; i < sz; i++) {
                TreeNode node = q.poll();
                cur.add(node.val);
                if (node.left != null) {
                    q.offer(node.left);
                }
                if (node.right != null) {
                    q.offer(node.right);
                }
            }
            ans.add(cur);
        }
        return ans;
    }

    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> ans = new LinkedList<>();
        Queue<TreeNode> q = new LinkedList<>();
        if (root == null) {
            return ans;
        }
        q.offer(root);
        int cur_leval = 0;
        while (!q.isEmpty()) {
            int sz = q.size();
            List<Integer> cur = new LinkedList<>();
            for (int i = 0; i < sz; i++) {
                TreeNode node = q.poll();
                cur.add(node.val);
                if (node.left != null) {
                    q.offer(node.left);
                }
                if (node.right != null) {
                    q.offer(node.right);
                }
            }
            cur_leval++;
            if (cur_leval % 2 == 0) {
                reverse(cur);
            }
            ans.add(cur);
        }
        return ans;
    }

    public void reverse(List<Integer> nums) {
        int start = 0, end = nums.size() - 1;
        while (start < end) {
            int temp = nums.get(start);
            nums.set(start, nums.get(end));
            nums.set(end, temp);
            start++;
            end--;
        }
    }

    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        LinkedList<List<Integer>> ans = new LinkedList<>();
        Queue<TreeNode> q = new LinkedList<>();
        if (root == null) {
            return ans;
        }
        q.offer(root);
        while (!q.isEmpty()) {
            int sz = q.size();
            List<Integer> cur = new LinkedList<>();
            for (int i = 0; i < sz; i++) {
                TreeNode node = q.poll();
                cur.add(node.val);
                if (node.left != null) {
                    q.offer(node.left);
                }
                if (node.right != null) {
                    q.offer(node.right);
                }
            }
            ans.addFirst(cur);
        }
        return ans;
    }

    public int minDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        int depth = 1;
        while (!q.isEmpty()) {
            int sz = q.size();
            for (int i = 0; i < sz; i++) {
                TreeNode cur = q.poll();
                if (cur.left == null && cur.right == null) {
                    return depth;
                }
                if (cur.left != null) {
                    q.offer(cur.left);
                }
                if (cur.right != null) {
                    q.offer(cur.right);
                }
            }
            depth++;
        }
        return -1;
    }

    public int numBusesToDestination(int[][] routes, int source, int target) {
        if (source == target) {
            return 0;
        }
        Queue<Integer> q = new LinkedList<>();
        q.offer(source);
        int changes = 0;
        HashSet<Integer> s = new HashSet<>();
        s.add(source);
        while (!q.isEmpty()) {
            int sz = q.size();
            for (int i = 0; i < sz; i++) {
                int cur = q.poll();
                if (cur == target) {
                    return changes;
                }
                for (int[] bus : routes) {
                    for (int stop : bus) {
                        if (stop == cur) {
                            for (int b : bus) {
                                if (!s.contains(b)) {
                                    q.offer(b);
                                    s.add(b);
                                }
                            }
                        }
                    }
                }
            }
            changes++;
        }
        return -1;
    }

    public int findCheapestPrice(int n, int[][] flights, int src, int dst, int K) {
        if (src == dst) {
            return 0;
        }
        int[][] graph = new int[n][];
        for (int[] flight : flights) {

        }
        Queue<Integer> q = new LinkedList<>();
        q.offer(src);
        int changes = 0;
        while (!q.isEmpty()) {
            int sz = q.size();
            for (int i = 0; i < sz; i++) {
                int cur = q.poll();

            }
        }
        return -1;
    }

    HashMap<TreeNode, TreeNode> mpOfTree = new HashMap<>();
    TreeNode src = new TreeNode(-1);

    public List<Integer> distanceK(TreeNode root, TreeNode target, int K) {
        List<Integer> ans = new LinkedList<>();
        if (root == null) {
            return ans;
        }
        treeTraverse(root, target);
        Queue<TreeNode> q = new LinkedList<>();
        HashSet<TreeNode> set = new HashSet<>();
        q.offer(src);
        set.add(src);
        int distance = 0;
        while (!q.isEmpty()) {
            int sz = q.size();
            for (int i = 0; i < sz; i++) {
                TreeNode cur = q.poll();
                if (distance == K) {
                    ans.add(cur.val);
                }
                if (cur.left != null && !set.contains(cur.left)) {
                    q.offer(cur.left);
                    set.add(cur.left);
                }
                if (cur.right != null && !set.contains(cur.right)) {
                    q.offer(cur.right);
                    set.add(cur.right);
                }
                if (mpOfTree.get(cur) != null && !set.contains(mpOfTree.get(cur))) {
                    q.offer(mpOfTree.get(cur));
                    set.add(mpOfTree.get(cur));
                }
            }
            distance++;
        }
        return ans;
    }

    public void treeTraverse(TreeNode root, TreeNode target) {
        if (root == null) {
            return;
        }
        if (root.left != null) {
            mpOfTree.put(root.left, root);
        }
        if (root.right != null) {
            mpOfTree.put(root.right, root);
        }
        if (root.val == target.val) {
            src = root;
        }
        treeTraverse(root.left, target);
        treeTraverse(root.right, target);
    }

    public int findBottomLeftValue(TreeNode root) {
        Queue<TreeNode> q = new LinkedList<>();
        List<Integer> ans = new LinkedList<>();
        q.offer(root);
        int num_level = 0;
        while (!q.isEmpty()) {
            int sz = q.size();
            for (int i = 0; i < sz; i++) {
                TreeNode cur = q.poll();
                ans.add(cur.val);
                if (cur.left != null) {
                    q.offer(cur.left);
                }
                if (cur.right != null) {
                    q.offer(cur.right);
                }
            }
            num_level++;
        }
        q.offer(root);
        int pre_nodes = 0;
        int cur_level = 0;
        while (!q.isEmpty()) {
            int sz = q.size();
            for (int i = 0; i < sz; i++) {
                TreeNode cur = q.poll();
                pre_nodes++;
                ans.add(cur.val);
                if (cur.left != null) {
                    q.offer(cur.left);
                }
                if (cur.right != null) {
                    q.offer(cur.right);
                }
            }
            cur_level++;
            if (cur_level == num_level - 1) {
                return ans.get(pre_nodes);
            }
        }
        return -1;
    }

    public int minimumJumps(int[] forbidden, int a, int b, int x) {
        HashSet<Integer> visited = new HashSet<>();
        visited.add(0);
        for (int f : forbidden) {
            visited.add(f);
        }
        Queue<Integer> q = new LinkedList<>();
        int start = 0;
        q.offer(start);
        int steps = 0;
        int right_bound = x + a;
        while (!q.isEmpty()) {
            int sz = q.size();
            for (int i = 0; i < sz; i++) {
                int cur = q.poll();
                if (cur == x) {
                    return steps;
                }
                int choice1 = cur + a, choice2 = cur - b;
                if (choice1 >= 0 && choice1 <= right_bound && !visited.contains(choice1)) {
                    q.offer(choice1);
                    visited.add(choice1);
                }
                if (choice2 >= 0 && choice2 <= right_bound && !visited.contains(choice2)) {
                    q.offer(choice2);
                    visited.add(choice2);
                }
            }
            steps++;
        }
        return -1;
    }

    public int minJumps(int[] arr) {
        int n = arr.length;
        boolean[] visited = new boolean[n];
        Queue<Integer> q = new LinkedList<>();//队列元素为坐标
        HashMap<Integer, List<Integer>> mp = new HashMap<>();
        for (int i = 0; i < n; i++) {
            if (!mp.containsKey(arr[i])) {
                mp.put(arr[i], new ArrayList<>());
                mp.get(arr[i]).add(i);
            } else {
                mp.get(arr[i]).add(i);
            }
        }
        int step = 0;
        q.offer(0);
        visited[0] = true;
        while (!q.isEmpty()) {
            int sz = q.size();
            for (int j = 0; j < sz; j++) {
                int cur = q.poll();
                if (cur == n - 1) {
                    return step;
                }
                if (cur - 1 >= 0 && cur - 1 < n && !visited[cur - 1]) {
                    q.offer(cur - 1);
                    visited[cur - 1] = true;
                }
                if (cur + 1 >= 0 && cur + 1 < n && !visited[cur + 1]) {
                    q.offer(cur + 1);
                    visited[cur + 1] = true;
                }
                for (int k : mp.get(arr[cur])) {
                    if (k != cur && !visited[k]) {
                        q.offer(k);
                    }
                }
            }
            step++;
        }
        return -1;
    }

    public List<String> watchedVideosByFriends(List<List<String>> watchedVideos, int[][] friends, int id, int level) {
        int n = watchedVideos.size();
        boolean[] visited = new boolean[n];
        Queue<Integer> q = new LinkedList<>();
        q.offer(id);
        visited[id] = true;
        int step = 0;
        int preorder = 0;//这里标记的是到所求level上一层结束时共有多少节点
        int start = 0, end = 0;
        while (!q.isEmpty()) {
            int sz = q.size();
            for (int i = 0; i < sz; i++) {
                int cur = q.poll();
                preorder++;
                for (int j : friends[cur]) {
                    if (!visited[j]) {
                        q.offer(j);
                        visited[j] = true;
                    }
                }
            }
            step++;
            if (step == level) {
                start = preorder;
            }
            if (step == level + 1) {
                end = preorder;
            }
        }
        HashMap<String, Integer> mp = new HashMap<>();
        for (int i = start; i < end; i++) {
            for (String s : watchedVideos.get(i)) {
                mp.put(s, mp.getOrDefault(s, 0) + 1);
            }
        }
        Comparator<Map.Entry<String, Integer>> c = new Comparator<Map.Entry<String, Integer>>() {
            @Override
            public int compare(Map.Entry<String, Integer> o1, Map.Entry<String, Integer> o2) {
                return o1.getValue().equals(o2.getValue()) ? o1.getKey().compareTo(o2.getKey()) : o1.getValue() - o2.getValue();
            }
        };
        List<String> ans = new LinkedList<>();
        for (String key : mp.keySet()) {
            ans.add(key);
        }
        return ans;
    }

    HashMap<Integer, Double> memory = new HashMap<>();

    public double myPow1(double x, int n) {
        if (x == 0.0) {
            return x;
        }
        if (n == 0) {
            return 1.0;
        }
        if (n == 1) {
            return x;
        }
        if (n == -1) {
            return 1 / x;
        }
        if (memory.containsKey(n)) {
            return memory.get(n);
        }
        double ans = 0.0;
        if (n % 2 == 0) {
            ans = myPow1(x, n / 2) * myPow1(x, n / 2);
            memory.put(n, ans);
            return ans;
        } else {
            if (n > 0) {
                ans = x * myPow1(x, n / 2) * myPow1(x, n / 2);
                memory.put(n, ans);
                return ans;
            } else {
                ans = (1 / x) * myPow1(x, n / 2) * myPow1(x, n / 2);
                memory.put(n, ans);
                return ans;
            }
        }
    }

    public int minEatingSpeed(int[] piles, int H) {
        int n = piles.length;
        Arrays.sort(piles);
        int max_speed = piles[n - 1];
        int start = 1, end = max_speed + 1;
        while (start < end) {
            int mid = start + (end - start) / 2;
            if (canFinish(mid, piles, H)) {
                end = mid;
            } else {
                start = mid + 1;
            }
        }
        return start;
    }

    public boolean canFinish(int speed, int[] piles, int H) {
        int hour_needed = 0;
        for (int p : piles) {
            if (speed > p) {
                hour_needed++;
            } else {
                if (p % speed == 0) {
                    hour_needed += p / speed;
                } else {
                    hour_needed += p / speed + 1;
                }
            }
        }
        if (hour_needed <= H) {
            return true;
        }
        return false;
    }

    public int shipWithinDays(int[] weights, int D) {
        int sumWeight = 0, maxWeight = -1;
        for (int w : weights) {
            sumWeight += w;
            maxWeight = Math.max(maxWeight, w);
        }
        int start = maxWeight, end = sumWeight;
        while (start < end) {
            int mid = start + (end - start) / 2;
            if (isAvailable(mid, weights, D)) {
                end = mid;
            } else {
                start = mid + 1;
            }
        }
        return start;
    }

    public boolean isAvailable(int cap, int[] weights, int D) {
        int carryToHere = 0, currentLeft = cap, shipment = 0;
        while (carryToHere < weights.length) {
            currentLeft -= weights[carryToHere++];
            if (currentLeft < 0) {
                carryToHere--;
                currentLeft = cap;
                shipment++;
            } else if (currentLeft == 0) {
                currentLeft = cap;
                shipment++;
            } else {
                if (carryToHere == weights.length) {
                    shipment++;
                }
            }
        }
        if (shipment <= D) {
            return true;
        }
        return false;
    }

    public int trap(int[] height) {
        int n = height.length;
        int[] leftmax = new int[n], rightmax = new int[n];
        for (int i = 1; i < n; i++) {
            leftmax[i] = Math.max(height[i - 1], leftmax[i - 1]);
        }
        for (int i = n - 2; i >= 0; i--) {
            rightmax[i] = Math.max(height[i + 1], rightmax[i + 1]);
        }
        int ans = 0;
        for (int i = 1; i < n - 1; i++) {
            ans += Math.max(Math.min(rightmax[i], leftmax[i]) - height[i], 0);
        }
        return ans;
    }

    public String removeDuplicateLetters(String s) {
        char[] arr = s.toCharArray();
        Arrays.sort(arr);
        int n = arr.length;
        if (n == 1 || n == 0) {
            return s;
        }
        int slow = 0, fast = 1;
        while (fast < n) {
            if (arr[fast] != arr[slow]) {
                arr[slow + 1] = arr[fast];
                slow++;
            }
            fast++;
        }
        String ans = "";
        for (int i = 0; i <= slow; i++) {
            ans += arr[i];
        }
        return ans;
    }

    int cardSum = 0;
    HashMap<int[], Integer> cardscore = new HashMap<>();

    public int maxScore(int[] cardPoints, int k) {
        if (k == 0) {
            return cardSum;
        }
        if (cardscore.containsKey(cardPoints)) {
            return cardscore.get(cardPoints);
        }
//        int ans= Math.max(cardPoints[0] + maxScore(Arrays.copyOfRange(cardPoints, 1, cardPoints.length), k - 1),
//                cardPoints[cardPoints.length - 1] + maxScore(Arrays.copyOfRange(cardPoints, 0, cardPoints.length - 1), k - 1));
        int sum1 = cardPoints[0] + maxScore(Arrays.copyOfRange(cardPoints, 1, cardPoints.length), k - 1);
        cardscore.put(Arrays.copyOfRange(cardPoints, 1, cardPoints.length), sum1);
        int sum2 = cardPoints[cardPoints.length - 1] + maxScore(Arrays.copyOfRange(cardPoints, 0, cardPoints.length - 1), k - 1);
        cardscore.put(Arrays.copyOfRange(cardPoints, 0, cardPoints.length - 1), sum2);
        return Math.max(sum1, sum2);
    }

    public List<String> removeInvalidParentheses(String s) {
        Stack<Character> st = new Stack<>();
        char[] arr = s.toCharArray();
        LinkedList<Character> arrlist = new LinkedList<>();
        for (char ch : arr) {
            arrlist.add(ch);
        }
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == '(') {
                st.push(arr[i]);
                arrlist.removeFirst();
            } else if (arr[i] == ')') {
                if (!st.isEmpty() && st.peek() == '(') {
                    st.pop();
                    arrlist.removeFirst();
                } else {
                    st.push(arr[i]);
                    arrlist.removeFirst();
                }
            } else {//本题中这里表示的是该字符为字母
                arrlist.removeFirst();
            }
        }
        int invalidLeft = 0, invalidRight = 0;
        while (!st.isEmpty()) {
            char temp = st.pop();
            if (temp == '(') {
                invalidLeft++;
            } else {
                invalidRight++;
            }
        }
        System.out.println("erereerere");
        backtrack(s, 0, invalidLeft, invalidRight);
        return Answer;
    }

    List<String> Answer = new LinkedList<>();

    public void backtrack(String s, int index, int invalidLeft, int invalidRight) {
        if (index == s.length() - invalidLeft - invalidRight) {//由于经过了删除，因此大概是这样
            if (invalidLeft == 0 && invalidRight == 0 && isValid(s)) {
                Answer.add(s);
            }
            return;
        }
        char temp = s.charAt(index);
        if (temp == '(') {
            s = s.substring(0, index) + s.substring(index + 1);
            backtrack(s, index, invalidLeft - 1, invalidRight);
            s = s.substring(0, index) + temp + s.substring(index);
        } else if (temp == ')') {
            s = s.substring(0, index) + s.substring(index + 1);
            backtrack(s, index, invalidLeft, invalidRight - 1);
            s = s.substring(0, index) + temp + s.substring(index);
        } else {
            backtrack(s, index + 1, invalidLeft, invalidRight);
        }
    }

    List<List<Integer>> paths = new LinkedList<>();

    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        if (root == null) {
            return new LinkedList<List<Integer>>();
        }
        backtrack(root, new LinkedList<>(), 0, targetSum);
        return paths;
    }

    public void backtrack(TreeNode root, LinkedList<Integer> path, int curSum, int targetSum) {
        if (root == null) {
            if (curSum == targetSum) {
                if (!paths.contains(path)) {
                    paths.add(new LinkedList<>(path));
                }
            }
            return;
        }
        curSum += root.val;
        path.add(root.val);
        backtrack(root.left, path, curSum, targetSum);
        backtrack(root.right, path, curSum, targetSum);
        curSum -= root.val;
        path.removeLast();
    }

    HashMap<Character, String> mymap = new HashMap<>();

    public List<String> letterCombinations(String digits) {

        mymap.put('2', "abc");
        mymap.put('3', "def");
        mymap.put('4', "ghi");
        mymap.put('5', "jkl");
        mymap.put('6', "mno");
        mymap.put('7', "pqrs");
        mymap.put('8', "tuv");
        mymap.put('9', "wxyz");
        backtrack(digits, "", 0);
        return combs;
    }

    List<String> combs = new LinkedList<>();

    public void backtrack(String digits, String ans, int current) {
        if (ans.length() == digits.length()) {
            combs.add(ans);
            return;
        }

        for (int j = 0; j < mymap.get(digits.charAt(current)).length(); j++) {
            ans += mymap.get(digits.charAt(current)).charAt(j);
            backtrack(digits, ans, current + 1);
            ans = ans.substring(0, current);
        }

    }

    public int sumNumbers(TreeNode root) {
        backtrack(root, 0);
        int ans = 0;
        for (int b : branch) {
            ans += b;
        }
        return ans;
    }

    List<Integer> branch = new LinkedList<>();

    public void backtrack(TreeNode root, int branchsum) {
        if (root == null) {
            return;
        }
        branchsum = branchsum * 10 + root.val;
        if (root.left == null && root.right == null) {
            branch.add(branchsum);
        }
        backtrack(root.left, branchsum);
        backtrack(root.right, branchsum);
        branchsum = (branchsum - root.val) / 10;
    }

    public boolean makesquare(int[] nums) {
        int sum = 0;
        for (int n : nums) {
            sum += n;
        }
        if (sum % 4 != 0) {
            return false;
        }
        int target = sum / 4;
        int[] piles = new int[4];
        return backtrack(nums, 0, piles, target);
    }

    boolean backtrack(int[] nums, int start, int[] piles, int target) {
        if (start == nums.length) {
            return piles[0] == piles[1] && piles[1] == piles[2] && piles[2] == piles[3];
        }
        for (int i = 0; i < 4; i++) {
            if (piles[i] + nums[start] > target) {
                continue;
            }
            piles[i] += nums[start];
            if (backtrack(nums, start + 1, piles, target)) {
                return true;
            }
            piles[i] -= nums[start];
        }
        return false;
    }


    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> ans = new LinkedList<>();
        if (root == null) {
            return ans;
        }
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            int sz = q.size();
            for (int i = 0; i < sz; i++) {
                TreeNode cur = q.poll();
                if (i == sz - 1) {
                    ans.add(cur.val);
                }
                if (cur.left != null) {
                    q.offer(cur.left);
                }
                if (cur.right != null) {
                    q.offer(cur.right);
                }
            }
        }
        return ans;
    }

    public boolean isValidBST(TreeNode root) {
        if (root == null) {
            return true;
        }
        if (root.left != null && root.left.val >= root.val) {
            return false;
        }
        if (root.right != null && root.right.val <= root.val) {
            return false;
        }
        if (root.right != null) {
            TreeNode rightRoot = root.right;
            while (rightRoot.left != null) {
                rightRoot = rightRoot.left;
            }
            if (rightRoot.val <= root.val) {
                return false;
            }
        }
        if (root.left != null) {
            TreeNode leftRoot = root.left;
            while (leftRoot.right != null) {
                leftRoot = leftRoot.right;
            }
            if (leftRoot.val >= root.val) {
                return false;
            }
        }
        return isValidBST(root.left) && isValidBST(root.right);
    }

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        backtrack(target, candidates, 0, 0);
        return sumAns;
    }

    List<List<Integer>> sumAns = new LinkedList<>();
    HashSet<List<Integer>> sumSet = new HashSet<>();
    LinkedList<Integer> valuable = new LinkedList<>();

    void backtrack(int target, int[] candidate, int curSum, int start) {
        if (curSum > target) {
            return;
        }
        if (curSum == target) {
            sumAns.add(new LinkedList<>(valuable));
            return;
        }
        for (int i = start; i < candidate.length; i++) {
            valuable.add(candidate[i]);
            backtrack(target, candidate, curSum + candidate[i], i);
            valuable.removeLast();
        }
    }

    public int sum(List<Integer> nums) {
        int ans = 0;
        for (int i : nums) {
            ans += i;
        }
        return ans;
    }

    public List<List<Integer>> findSubsequences(int[] nums) {
        Backtrack(nums, new LinkedList<>(), 0);
        return subs;
    }

    List<List<Integer>> subs = new LinkedList<>();
    HashSet<List<Integer>> myset = new HashSet<>();

    void Backtrack(int[] nums, LinkedList<Integer> item, int start) {
        if (start == nums.length) {//这里需要处理一下有返回结果的东西
            if (item.size() >= 2 && !myset.contains(item)) {
                subs.add(new LinkedList<>(item));
                myset.add(item);
            }
            return;
        }
        if (item.size() >= 2 && !myset.contains(item)) {
            subs.add(new LinkedList<>(item));
            myset.add(item);
        }
        for (int i = start; i < nums.length; i++) {
            if (item.size() > 0 && nums[i] < item.getLast()) {
                continue;
            }
            item.add(nums[i]);
            Backtrack(nums, item, i + 1);
            item.removeLast();
        }
    }

    public String longestPalindrome(String s) {
        int n = s.length();
        if (n == 0) {
            return "";
        }
        boolean[][] dp = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            dp[i][i] = true;
            if (i != n - 1) {
                dp[i + 1][i] = true;
            }
        }
        int maxDist = 1;
        int[] index = new int[]{0, 0};
        for (int i = n - 2; i >= 0; i--) {
            for (int j = i + 1; j < n; j++) {
                dp[i][j] = dp[i + 1][j - 1] & s.charAt(i) == s.charAt(j);
                if (dp[i][j] && j - i + 1 > maxDist) {
                    maxDist = j - i + 1;
                    index[0] = i;
                    index[1] = j;
                }
            }
        }
        return s.substring(index[0], index[1] + 1);
    }

    public int numDecodings(String s) {
        int n = s.length();
        if (n == 0 || s.charAt(0) == '0') {
            return 0;
        }
        int[] dp = new int[n];
        for (int i = 0; i < n; i++) {
            if (s.charAt(i) == '0') {
                dp[i] = 0;
            } else {
                dp[i] = 1;
            }
        }
        for (int i = 1; i < n; i++) {
            if (s.charAt(i) == '0') {
                if (s.charAt(i - 1) == '0') {
                    return 0;
                }
                if ((s.charAt(i - 1) - '0') * 10 > 26) {
                    dp[i] = 0;
                } else {
                    if (i == 1) {
                        dp[i] = 1;
                    } else {
                        dp[i] = dp[i - 2];
                    }
                }
            } else {
                if (s.charAt(i - 1) == '0') {
                    dp[i] = dp[i - 1];
                } else {
                    if ((s.charAt(i - 1) - '0') * 10 + s.charAt(i) - '0' > 26) {
                        dp[i] = dp[i - 1];
                    } else {
                        if (i > 1) {
                            dp[i] = dp[i - 1] + dp[i - 2];//这里可能过界}
                        } else {
                            dp[i] = 2;
                        }
                    }
                }
            }
        }
        return dp[n - 1];
    }

    public int minimumTotal(List<List<Integer>> triangle) {
        Backtrack(triangle, 0, 0);
        int max = Integer.MIN_VALUE;
        for (int i : triSum) {
            max = Math.max(max, i);
        }
        return max;
    }

    List<Integer> triSum = new LinkedList<>();

    void Backtrack(List<List<Integer>> trangle, int index, int sum) {
        if (index == trangle.size()) {
            triSum.add(sum);
            return;
        }
        sum += trangle.get(index).get(index);
        Backtrack(trangle, index + 1, sum);
        sum -= trangle.get(index).get(index);
        sum += trangle.get(index).get(index + 1);
        Backtrack(trangle, index + 1, sum);
        sum -= trangle.get(index).get(index + 1);
    }

    public int maxProduct(int[] nums) {
        int n = nums.length;
        int[][] dp = new int[n][2];
        dp[0][0] = nums[0];
        int max = Integer.MIN_VALUE;
        for (int i = 1; i < n; i++) {
            dp[i][0] = myMax(nums[i], dp[i - 1][0] * nums[i], dp[i - 1][1] * nums[i]);
            dp[i][1] = myMin(nums[i], dp[i - 1][0] * nums[i], dp[i - 1][1] * nums[i]);
            max = Math.max(max, dp[i][0]);
        }
        return max;
    }

    public int myMax(int a, int b, int c) {
        return Math.max(a, Math.max(b, c));
    }

    public int myMin(int a, int b, int c) {
        return Math.min(a, Math.min(b, c));
    }

    public int findNumberOfLIS(int[] nums) {
        int n = nums.length;
        int[] dp = new int[n];
        int[] cnt = new int[n];
        Arrays.fill(dp, 1);
        Arrays.fill(cnt, 1);
        int maxLength = 0;
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i]) {
                    if (dp[j] + 1 > dp[i]) {
                        dp[i] = dp[j] + 1;
                        cnt[i] = cnt[j];
                        maxLength = Math.max(maxLength, dp[i]);
                    } else if (dp[j] + 1 == dp[i]) {
                        cnt[i] += cnt[j];
                    }
                }
            }
        }
        int ans = 0;
        for (int i = 0; i < n; i++) {
            if (dp[i] == maxLength) {
                ans += cnt[i];
            }
        }
        return ans;
    }

    public boolean canCross(int[] stones) {
        int n = stones.length;
        if (n <= 1) {
            return true;
        }
        if (stones[1] - stones[0] > 1) {
            return false;
        }

        boolean[] dp = new boolean[n];
        dp[0] = true;
        List<Integer> stoneList = new LinkedList<>();
        for (int s : stones) {
            stoneList.add(s);
        }
        return BackTrack(stoneList, 1, 0);
    }

    HashMap<Integer, Boolean> stoneMap = new HashMap<>();
    boolean stoneAns = false;

    boolean BackTrack(List<Integer> stones, int k, int cur) {
        System.out.println("cur is " + cur);
        if (cur == stones.size() - 1) {
            return true;
        }
        if (stoneMap.containsKey(cur)) {
            return stoneMap.get(cur);
        }
        if (!stones.contains(stones.get(cur) + k - 1) && !stones.contains(stones.get(cur) + k)
                && !stones.contains(stones.get(cur) + k + 1)) {
            return false;
        }
        for (int i = cur + 1; i < stones.size(); i++) {
            if (stones.get(i) - stones.get(cur) == k) {
                stoneAns |= BackTrack(stones, k, i);
            }
            if (stones.get(i) - stones.get(cur) == k - 1) {
                stoneAns |= BackTrack(stones, k - 1, i);
            }
            if (stones.get(i) - stones.get(cur) == k + 1) {
                stoneAns |= BackTrack(stones, k + 1, i);
            }
        }
        stoneMap.put(cur, stoneAns);
        return stoneAns;
    }

    public int deleteAndEarn(int[] nums) {
        int[] bucket = new int[10001];
        int maxValue = -1;
        for (int n : nums) {
            bucket[n] += n;
            maxValue = Math.max(maxValue, n);
        }
        int[] dp = new int[10001];
        dp[0] = 0;
        dp[1] = bucket[1];
        for (int i = 2; i <= maxValue; i++) {
            dp[i] = Math.max(dp[i - 2] + bucket[i], dp[i - 1]);
        }
        return dp[maxValue];
    }

    public List<Integer> largestValues(TreeNode root) {
        List<Integer> ans = new LinkedList<>();
        if (root == null) {
            return ans;
        }
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            int sz = q.size();
            int maxValue = Integer.MIN_VALUE;
            for (int i = 0; i < sz; i++) {
                TreeNode cur = q.poll();
                maxValue = Math.max(cur.val, maxValue);
                if (cur.left != null) {
                    q.offer(cur.left);
                }
                if (cur.right != null) {
                    q.offer(cur.right);
                }
            }
            ans.add(maxValue);
        }
        return ans;
    }

    public boolean isCompleteTree(TreeNode root) {
        if (root == null) {
            return true;
        }
        if (root.left == null && root.right == null) {
            return true;
        }
        if (root.right != null && root.left == null) {
            return false;
        }
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            int sz = q.size();
            for (int i = 0; i < sz; i++) {
                TreeNode cur = q.poll();
                if (i < sz - 1) {
                    if (cur.left == null || cur.right == null) {
                        return false;
                    }
                } else {
                    if (cur.left == null && cur.right != null) {
                        return false;
                    }
                }
                if (cur.left != null) {
                    q.offer(cur.left);
                }
                if (cur.right != null) {
                    q.offer(cur.right);
                }
            }
        }
        return true;
    }

    public String minRemoveToMakeValid(String s) {
        Stack<Character> st = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                st.push(s.charAt(i));
            } else if (s.charAt(i) == ')') {
                if (!st.isEmpty() && st.peek() == '(') {
                    st.pop();
                } else {
                    st.push(s.charAt(i));
                }
            }
        }
        while (!st.isEmpty()) {
            char ch = st.pop();
            for (int i = 0; i < s.length(); i++) {
                if (s.charAt(i) == ch) {
                    s = s.substring(0, i) + s.substring(i + 1);
                    break;
                }
            }
        }
        return s;
    }


    HashMap<String, Integer> acc = new HashMap<>();//这里记录的是这个邮箱第一次出现的坐标
    HashMap<String, List<Integer>> showUp = new HashMap<>();

    public List<List<String>> accountsMerge(List<List<String>> accounts) {
        int n = accounts.size();
        UnionFind uf = new UnionFind(n);
        List<List<String>> ans = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            for (int j = 1; j < accounts.get(i).size(); j++) {
                if (acc.containsKey(accounts.get(i).get(j))) {
                    uf.union(i, acc.get(accounts.get(i).get(j)));
                } else {
                    acc.put(accounts.get(i).get(j), i);
                }
            }
        }
        for (int i = 0; i < n; i++) {
            List<String> tmp = new LinkedList<>();
            tmp.add(accounts.get(i).get(0));
            ans.add(tmp);
        }
        for (int i = 0; i < n; i++) {
            if (i == uf.parent[i]) {
                for (int j = 1; j < accounts.get(i).size(); j++) {
                    ans.get(i).add(accounts.get(i).get(j));
                }
            } else {
                for (int j = 1; j < accounts.get(i).size(); j++) {
                    if (!ans.get(uf.parent[i]).contains(accounts.get(i).get(j))) {
                        ans.get(uf.parent[i]).add(accounts.get(i).get(j));
                    }
                }
            }
        }
        ans.removeIf(s -> s.size() == 1);
        return ans;
    }

    public int[] numsSameConsecDiff(int n, int k) {
        for (int i = 1; i < 10; i++) {
            BackTrack(n, k, 1, i);
        }
        int[] ans = new int[numsAns.size()];
        int index = 0;
        for (int i : numsAns) {
            ans[index++] = i;
        }
        return ans;
    }

    LinkedList<Integer> numsTrack = new LinkedList<>();
    List<Integer> numsAns = new LinkedList<>();

    void BackTrack(int n, int k, int curLen, int curBit) {
        if (curBit > 9 || curBit < 0) {
            return;
        }
        if (curLen > n) {
            int ans = 0;
            for (int i = 0; i < numsTrack.size(); i++) {
                ans = ans * 10 + numsTrack.get(i);
            }
            numsAns.add(ans);
            return;
        }
        numsTrack.add(curBit);
        BackTrack(n, k, curLen + 1, curBit + k);
        numsTrack.removeLast();
        numsTrack.add(curBit);
        BackTrack(n, k, curLen + 1, curBit - k);
        numsTrack.removeLast();
    }

    public int minSwapsCouples(int[] row) {
        int len = row.length;
        int N = len / 2;
        UnionFind uf = new UnionFind(N);
        for (int i = 0; i < len; i += 2) {
            uf.union(row[i] / 2, row[i + 1] / 2);
        }
        return N - uf.count();
    }

    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode p1 = l1, p2 = l2;
        ListNode head = new ListNode((p1.val + p2.val) % 10);
        int upBit = (p1.val + p2.val) / 10;
        ListNode cur = head;
        p1 = p1.next;
        p2 = p2.next;
        while (p1 != null && p2 != null) {
            int curSum = p1.val + p2.val + upBit;
            upBit = curSum / 10;
            int curBit = (curSum) % 10;
            ListNode node = new ListNode(curBit);
            cur.next = node;
            cur = cur.next;
            p1 = p1.next;
            p2 = p2.next;
        }
        while (p1 != null) {
            int curSum = p1.val + upBit;
            upBit = curSum / 10;
            int curBit = (curSum) % 10;
            ListNode node = new ListNode(curBit);
            cur.next = node;
            cur = cur.next;
            p1 = p1.next;
        }
        while (p2 != null) {
            int curSum = p2.val + upBit;
            upBit = curSum / 10;
            int curBit = (curSum) % 10;
            ListNode node = new ListNode(curBit);
            cur.next = node;
            cur = cur.next;
            p2 = p2.next;
        }
        if (upBit != 0) {
            ListNode end = new ListNode(upBit);
            cur.next = end;
        }
        return head;
    }

    public List<List<Integer>> fourSum(int[] nums, int target) {
        Arrays.sort(nums);
        fourBackTrack(nums, 0, 0, 0, target);
        for (List<Integer> s : fourAnsSet) {
            fourAns.add(s);
        }
        return fourAns;
    }

    List<List<Integer>> fourAns = new LinkedList<>();
    LinkedList<Integer> fourTrack = new LinkedList<>();
    HashSet<Integer> fourSet = new HashSet<>();
    HashSet<List<Integer>> fourAnsSet = new HashSet<>();

    void fourBackTrack(int[] nums, int curSize, int curSum, int curIndex, int target) {
        if (curSize == 4) {
            if (curSum == target) {
                fourAnsSet.add(new LinkedList<>(fourTrack));
            }
            return;
        }
        if (curIndex >= nums.length) {
            return;
        }
        for (int i = curIndex; i < nums.length; i++) {
            if (!fourSet.contains(i)) {
                fourTrack.add(nums[i]);
                fourSet.add(i);
                fourBackTrack(nums, curSize + 1, curSum + nums[i], i, target);
                fourTrack.removeLast();
                fourSet.remove(i);
            }
        }
    }

    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode p = head, q = head, pre = p;
        for (int i = 0; i < n; i++) {
            q = q.next;
        }
        while (q != null) {
            pre = p;
            p = p.next;
            q = q.next;
        }
        pre.next = p.next;
        if (p == head) {
            return head.next;
        }
        return head;
    }

    public int divide(int dividend, int divisor) {
        int ans = 0;
        int cache1 = dividend, cache2 = divisor;
        if (Math.abs(dividend) < Math.abs(divisor) && dividend != Integer.MIN_VALUE) {
            return 0;
        }
        if (dividend > 0) {
            if (divisor > 0) {
                while (Math.abs(dividend) >= Math.abs(divisor)) {
                    dividend -= divisor;
                    ans++;
                }
            } else {
                while (Math.abs(dividend) >= Math.abs(divisor)) {
                    dividend += divisor;
                    ans--;
                }
            }
        } else {
            if (divisor < 0) {
                while (Math.abs(dividend) >= Math.abs(divisor)) {
                    dividend -= divisor;
                    ans++;
                }
            } else {
                while (Math.abs(dividend) >= Math.abs(divisor)) {
                    dividend += divisor;
                    ans--;
                }
            }
        }
        if (ans == 0) {
            if ((cache1 > 0 && cache2 > 0) || (cache1 < 0 && cache2 < 0)) {
                return Integer.MAX_VALUE;
            } else {
                return Integer.MIN_VALUE;
            }
        }
        return ans;
    }

    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode p = l1, q = l2;
        ListNode head = new ListNode(0);
        ListNode cur = head;
        while (p != null && q != null) {
            if (p.val < q.val) {
                ListNode tmp = new ListNode(p.val);
                cur.next = tmp;
                cur = cur.next;
                p = p.next;
            } else {
                ListNode tmp = new ListNode(q.val);
                cur.next = tmp;
                cur = cur.next;
                q = q.next;
            }
        }
        while (p != null) {
            ListNode tmp = new ListNode(p.val);
            cur.next = tmp;
            cur = cur.next;
            p = p.next;
        }
        while (q != null) {
            ListNode tmp = new ListNode(q.val);
            cur.next = tmp;
            cur = cur.next;
            q = q.next;
        }
        return head.next;
    }

    public List<List<Integer>> permuteUnique(int[] nums) {
        permuteBackTrack(nums);
        List<List<Integer>> ans = new LinkedList<>();
        for (List<Integer> p : permuteAns) {
            ans.add(p);
        }
        return ans;
    }

    LinkedList<Integer> permuteTrack = new LinkedList<>();
    HashSet<List<Integer>> permuteAns = new HashSet<>();
    HashSet<Integer> permuteSet = new HashSet<>();

    void permuteBackTrack(int[] nums) {
        if (permuteTrack.size() == nums.length) {
            permuteAns.add(new LinkedList<>(permuteTrack));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (permuteSet.contains(i)) {
                continue;
            }
            permuteTrack.add(nums[i]);
            permuteSet.add(i);
            permuteBackTrack(nums);
            permuteSet.remove(i);
            permuteTrack.removeLast();
        }
    }

    public int shortestPath(int[][] grid, int k) {
        int m = grid.length, n = grid[0].length;
        boolean[][][] visited = new boolean[m][n][k+1];
        Queue<int[]> q = new LinkedList<>();
        q.offer(new int[]{0, 0, k});
        visited[0][0][k] = true;
        int[] dx = {1, 0, -1, 0}, dy = {0, 1, 0, -1};
        int step = 0;
        while (!q.isEmpty()) {
            int sz = q.size();
            for (int i = 0; i < sz; i++) {
                int[] cur = q.poll();
//                visited[cur[0]][cur[1]]=true;
                if (cur[0] == m - 1 && cur[1] == n - 1 && cur[2] >= 0) {
                    return step;
                }
                for (int j = 0; j < 4; j++) {
                    int[] position = new int[]{cur[0], cur[1], cur[2]};
                    position[0] += dx[j];
                    position[1] += dy[j];
                    if (position[0] >= 0 && position[0] < m && position[1] >= 0 && position[1] < n &&
                            !visited[position[0]][position[1]][position[2]]) {
                        if (grid[position[0]][position[1]] == 0) {
                            q.offer(position);
                            visited[position[0]][position[1]][position[2]] = true;
                        } else {
                            if(position[2]>0) {
                                position[2]--;
                                q.offer(position);
                                visited[position[0]][position[1]][position[2]] = true;
                            }
                        }
                    }
                }
            }
            step++;
        }
        return -1;
    }

    public static void main(String[] args) {
        Solution tt = new Solution();
        Scanner in=new Scanner(System.in);
        int m=2,n=2;
        char[][]arr=new char[m][n];
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                arr[i][j]=in.next().charAt(0);
                System.out.println(arr[i][j]);
            }
        }
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                System.out.println(arr[i][j]);
            }
        }
    }
}
