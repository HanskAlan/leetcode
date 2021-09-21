import java.util.*;
import java.lang.*;

public class Practice {
    public boolean isBalanced(TreeNode root) {
        if (root == null) {
            return true;
        }
        return Math.abs(height(root.left) - height(root.right)) <= 1 && isBalanced(root.left) && isBalanced(root.right);
    }

    public int height(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return Math.max(1 + height(root.left), 1 + height(root.right));
    }


    public void reorderList(ListNode head) {
        if (head == null || head.next == null) {
            return;
        }
        ListNode p = head;
        while (p.next != null) {
            ListNode nxt = p.next;
            ListNode pre = p;
            ListNode probe = p.next;
            while (probe.next != null) {
                pre = pre.next;
                probe = probe.next;
            }
            p.next = probe;
            if (probe != nxt) {
                probe.next = nxt;
                pre.next = null;
            }
            p = nxt;
            nxt = nxt.next;

        }
    }

    public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
        wordList.add(beginWord);
        int n = wordList.size();
        boolean[] visited = new boolean[n];
        visited[n - 1] = true;
        Queue<Integer> q = new LinkedList<>();
        HashMap<Integer, List<Integer>> adj = new HashMap<>();
        q.offer(n - 1);
        int step = 1;
        boolean flag = false;
        int target = 0;
        int level = 0;
        while (!q.isEmpty()) {
            int sz = q.size();
            for (int i = 0; i < sz; i++) {
                int cur = q.poll();
                if (wordList.get(cur).equals(endWord)) {
                    flag = true;
                    target = cur;
                    level = step;
                }
                for (int j = 0; j < n; j++) {
                    if (isOneBitDiff(wordList.get(cur), wordList.get(j))) {
                        q.offer(j);
                        if (!adj.containsKey(cur)) {
                            adj.put(cur, new LinkedList<>());
                        }
                        adj.get(cur).add(j);
                    }
                }
            }
            step++;
        }
        if (!flag) {
            return new LinkedList<>();
        }
        Arrays.fill(visited, false);
        dfs(adj, n - 1, target, step, new LinkedList<>(), wordList, visited);
        return ans;
    }

    List<List<String>> ans = new LinkedList<>();

    void dfs(HashMap<Integer, List<Integer>> adj, int cur, int target,
             int level, LinkedList<String> track, List<String> wordList, boolean[] visited) {
        if (level == 0) {
            if (track.getLast().equals(wordList.get(target))) {
                ans.add(new LinkedList<>(track));
            }
            return;
        }
        for (int i : adj.get(cur)) {
            if (!visited[i]) {
                track.add(wordList.get(i));
                visited[i] = true;
                if (adj.containsKey(i)) {
                    dfs(adj, i, target, level - 1, track, wordList, visited);
                }
                visited[i] = false;
                track.removeLast();
            }
        }
    }

    public boolean isOneBitDiff(String a, String b) {
        if (a.length() != b.length()) {
            return false;
        }
        int diffnum = 0;
        for (int i = 0; i < a.length(); i++) {
            if (a.charAt(i) != b.charAt(i)) {
                diffnum++;
                if (diffnum > 1) {
                    return false;
                }
            }
        }
        return diffnum == 1;
    }

    public int rangeBitwiseAnd(int m, int n) {
        int mask = 1 << 30;
        int ans = 0;
        while (mask > 0 && (m & mask) == (n & mask)) {
            ans |= (m & mask);
            mask >>= 1;
        }
        return ans;
    }

    public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        int left = 0, right = k;
        for (int i = 0; i < right; i++) {
            for (int j = i + 1; j < right; j++) {
                if (isOverflow(nums[i], nums[j])) {
                    return false;
                }
                if (Math.abs(nums[i] - nums[j]) <= t) {
                    return true;
                }

            }
        }
        int n = nums.length;
        while (right < n) {
            for (int i = left; i < right; i++) {
                if (isOverflow(nums[right], nums[i])) {
                    return false;
                }
                if (Math.abs(nums[right] - nums[i]) <= t) {
                    return true;
                }
            }
            left++;
            right++;
        }
        return false;
    }

    public boolean isOverflow(int x, int y) {
        int r = x - y;
        // HD 2-12 Overflow iff the arguments have different signs and
        // the sign of the result is different than the sign of x
        if (((x ^ y) & (x ^ r)) < 0) {
            return true;
        }
        return false;
    }

    public int countNodes(TreeNode root) {
        if (root == null) {
            return 0;
        }
        TreeNode lp = root.left;
        int height = 1;
        while (lp != null) {
            height++;
            lp = lp.left;
        }
        int lr = 2;
        if (height > 1) {
            TreeNode leftR = root.left.right;
            while (leftR != null) {
                lr++;
                leftR = leftR.right;
            }
        }
        if (lr == height) {
            return 1 + (int) (Math.pow(2, height - 1) - 1) + recCount(root.right);
        } else {
            return 1 + recCount(root.left) + (int) (Math.pow(2, height - 2) - 1);
        }
    }

    public int recCount(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return 1 + recCount(root.left) + recCount(root.right);
    }

    public List<String> binaryTreePaths(TreeNode root) {
        dfs(root);
        List<String> ans = new LinkedList<>();
        for (List<Integer> l : tracks) {
            String tmp = "";
            for (int i = 0; i < l.size(); i++) {
                tmp = tmp + l.get(i);
                if (i != l.size() - 1) {
                    tmp += "->";
                }
            }
            ans.add(tmp);
        }
        return ans;
    }

    List<List<Integer>> tracks = new LinkedList<>();
    LinkedList<Integer> track = new LinkedList<>();

    void dfs(TreeNode root) {
        if (root.left == null && root.right == null) {
            track.add(root.val);
            tracks.add(new LinkedList<>(track));
            return;
        }
        track.add(root.val);
        if (root.left != null) {
            dfs(root.left);
            track.removeLast();
        }
        if (root.right != null) {
            dfs(root.right);
            track.removeLast();
        }
    }

    public List<Integer> grayCode(int n) {
        l0.add(0);
        l1.add(0);
        l1.add(1);
        return backtrack(n);
    }

    LinkedList<Integer> l0 = new LinkedList<>();
    LinkedList<Integer> l1 = new LinkedList<>();

    LinkedList<Integer> backtrack(int n) {
        if (n == 0) {
            return l0;
        }
        if (n == 1) {
            return l1;
        }
        LinkedList<Integer> last = backtrack(n - 1);
        LinkedList<Integer> ans = new LinkedList<>(last);
        while (last.size() != 0) {
            int cur = last.getLast();
            last.removeLast();
            int newcur = cur ^ (int) (Math.pow(2, n - 1));
            ans.add(newcur);
        }
        return ans;
    }

    public String shortestPalindrome(String s) {
        int n = s.length();
        int pivot = -1;
        for (int i = n; i >= 0; i--) {
            if (isPalindrome(s.substring(0, i))) {
                pivot = i;
                break;
            }
        }
        return reverse(s.substring(pivot)) + s;
    }

    String reverse(String s) {
        String ans = "";
        for (int i = s.length() - 1; i >= 0; i--) {
            ans = ans + s.charAt(i);
        }
        return ans;
    }

    boolean isPalindrome(String s) {
        int n = s.length();
        int mid = n / 2;
        for (int i = 0; i < mid; i++) {
            if (s.charAt(i) != s.charAt(n - 1 - i)) {
                return false;
            }
        }
        return true;
    }

    public int minimumTotal(List<List<Integer>> triangle) {
        int n = triangle.size();
        int[][] dp = new int[n][n];
        dp[0][0] = triangle.get(0).get(0);
//        dp[1][0]=dp[0][0]+triangle.get(1).get(0);
//        dp[1][1]=dp[0][0]+triangle.get()
        for (int i = 1; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                if (j - 1 < 0) {
                    dp[i][j] = dp[i - 1][j] + triangle.get(i).get(j);
                } else if (j == i) {
                    dp[i][j] = dp[i - 1][j - 1] + triangle.get(i).get(j);
                } else {
                    dp[i][j] = Math.min(dp[i - 1][j], dp[i - 1][j - 1]) + triangle.get(i).get(j);
                }
            }
        }
        int ans = Integer.MAX_VALUE;
        for (int i : dp[n - 1]) {
            ans = Math.min(ans, i);
        }
        return ans;
    }

    public int findMin(int[] nums) {
        int n = nums.length;
        if (nums[n - 1] >= nums[0]) {
            return nums[0];
        }
        int left = 0, right = n - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == nums[left]) {
                return Math.min(nums[left], nums[right]);
            }
            if (nums[mid] > nums[left]) {
                if (nums[mid] > nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            } else if (nums[mid] < nums[left]) {
                left++;
            }
        }
        return nums[left];
    }

    public int minOps(String s1, String s2) {
        int m = s1.length(), n = s2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 0; i <= m; i++) {
            dp[i][0] = i;
        }
        for (int i = 0; i <= n; i++) {
            dp[0][i] = i;
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = getMin(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1;
                }
            }
        }
        return dp[m][n];
    }

    public int getMin(int a, int b, int c) {
        return Math.min(a, Math.min(b, c));
    }

    public boolean canJump(int[] nums) {
        int n = nums.length;
        int farthest = 0;
        for (int i = 0; i < n - 1; i++) {
            if (i <= farthest) {
                farthest = Math.max(farthest, i + nums[i]);
            }
        }
        return farthest >= n - 1;
    }

    public String reverseVowels(String s) {
        HashSet<Character> vowels = new HashSet<>();
        char[] arr = {'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'};
        for (char c : arr) {
            vowels.add(c);
        }
        char[] ss = s.toCharArray();
        int left = 0, right = ss.length - 1;
        while (left < right) {
            while (left < right && !vowels.contains(ss[left])) {
                left++;
            }
            while (left < right && !vowels.contains(ss[right])) {
                right--;
            }
            char temp = ss[left];
            ss[left] = ss[right];
            ss[right] = temp;
            left++;
            right--;
        }
        return new String(ss);
    }

    public TreeNode invertTree(TreeNode root) {
        if (root == null) {
            return null;
        }
        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;
        root.left = invertTree(root.left);
        root.right = invertTree(root.right);
        return root;
    }

    public int hammingDistance(int x, int y) {
        int target = x ^ y;
        int m = 1;
        int dist = 0;
        while (m != 0) {
            if ((m & target) != 0) {
                dist++;
            }
            m <<= 1;
        }
        return dist;
    }

    public int myAtoi(String s) {
        int n = s.length();
        if (n == 0) {
            return 0;
        }
        int start = 0;
        for (int i = 0; i < n; i++) {
            if (s.charAt(i) == '-' || s.charAt(i) == '+' || Character.isDigit(s.charAt(i))) {
                start = i;
                break;
            }
        }
        for (int i = 0; i < start; i++) {
            if (s.charAt(i) != ' ') {
                return 0;
            }
        }
        boolean isNegative = false;
        if (s.charAt(start) == '-') {
            isNegative = true;
            start++;
        } else if (s.charAt(start) == '+') {
            isNegative = false;
            start++;
        }
        int end = -1;
        for (int i = start; i < n; i++) {
            if (!Character.isDigit(s.charAt(i))) {
                end = i;
                break;
            }
        }
        if (end != -1) {
            s = s.substring(start, end);
        } else {
            s = s.substring(start);
        }
        n = s.length();
        int ans = 0;
        for (int i = 0; i < n; i++) {
            ans = ans * 10 + (s.charAt(i) - '0');
            if (i != n - 1 && (ans > Integer.MAX_VALUE / 10 ||
                    (ans == Integer.MAX_VALUE / 10 && (s.charAt(i + 1) - '0') > Integer.MAX_VALUE % 10))) {
                if (isNegative) {
                    return Integer.MIN_VALUE;
                } else {
                    return Integer.MAX_VALUE;
                }
            }
        }
        int flag = isNegative ? -1 : 1;
        return ans * flag;
    }

    public ListNode mergeKLists(ListNode[] lists) {
        int n = lists.length;
        List<Integer> l = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            ListNode p = lists[i];
            while (p != null) {
                l.add(p.val);
                p = p.next;
            }
        }
        Collections.sort(l);
        ListNode head = new ListNode(-1);
        ListNode p = head;
        for (int i = 0; i < l.size(); i++) {
            ListNode newNode = new ListNode(l.get(i));
            p.next = newNode;
            p = p.next;
        }
        return head.next;
    }

    public int calculate(String s) {
        int n = s.length();
        Stack<Integer> st = new Stack<>();
        int end = n - 1;
        for (int i = n - 1; i >= 0; i--) {
            if (s.charAt(i) != ' ') {
                end = i;
                break;
            }
        }
        s = s.substring(0, end + 1);
        n = s.length();
        int num = 0;
        char op = '+';
        for (int i = 0; i < n; i++) {
            char currChar = s.charAt(i);
            if (currChar == ' ') {
                continue;
            }
            if (i < n - 1 && Character.isDigit(currChar)) {
                num = num * 10 + (currChar - '0');
            } else {
                if (i == n - 1) {
                    num = num * 10 + (currChar - '0');
                }
                switch (op) {
                    case '+':
                        st.push(num);
                        break;
                    case '-':
                        st.push(-num);
                        break;
                    case '*':
                        st.push(st.pop() * num);
                        break;
                    case '/':
                        st.push(st.pop() / num);
                        break;
                }
                op = currChar;
                num = 0;
            }
        }
        int ans = 0;
        while (!st.isEmpty()) {
            ans += st.pop();
        }
        return ans;
    }

    public int findDuplicate(int[] nums) {
        int n = nums.length;
        int[] table = new int[n];
        for (int i = 0; i < n; i++) {
            table[nums[i]]++;
            if (table[nums[i]] > 1) {
                return nums[i];
            }
        }
        return 0;
    }

    public int numDistinct(String s, String t) {
        int m = s.length(), n = t.length();
        if (n > m) {
            return 0;
        }
        int[][] dp = new int[m + 1][n];
        for (int i = 1; i <= m; i++) {
            dp[i][0] = dp[i - 1][0] + (s.charAt(i - 1) == t.charAt(0) ? 1 : 0);
        }
        for (int i = 1; i < n; i++) {
            for (int j = i; j < m; j++) {
                dp[j + 1][i] = (s.charAt(j) == t.charAt(i) ? dp[j][i - 1] : 0) + dp[j][i];
            }
        }
        return dp[m][n - 1];
    }

    public boolean isValidSerialization(String preorder) {
        String[] nodes = preorder.split(",");
        int n = nodes.length;
        Stack<Integer> st = new Stack<>();
        if (!nodes[0].equals("#")) {
            st.push(2);
        }
        for (int i = 1; i < n; i++) {
            if (nodes[i].equals("#")) {
                if (st.isEmpty()) {
                    return false;
                }
                st.push(st.pop() - 1);
                if (st.peek() == 0) {
                    st.pop();
                }
            } else {
                if (st.isEmpty()) {
                    return false;
                }
                st.push(st.pop() - 1);
                if (st.peek() == 0) {
                    st.pop();
                }
                st.push(2);
            }
        }
        return st.isEmpty();
    }

    public boolean canCross(int[] stones) {
        for (int st : stones) {
            s.add(st);
        }
        if (stones[1] != 1) {
            return false;
        }
        return backtrack(stones, 1, 0);
    }

    HashSet<Integer> s = new HashSet<>();//记录所有石头
    HashMap<String, Boolean> memo = new HashMap<>();

    boolean backtrack(int[] stones, int k, int curStone) {
        if (curStone == stones[stones.length - 1]) {
            return true;
        }
        if (k <= 0) {
            return false;
        }
        String key = curStone + "," + k;
        if (memo.containsKey(key)) {
            return memo.get(key);
        }
        if (s.contains(curStone)) {
            if (backtrack(stones, k - 1, curStone + k - 1) || backtrack(stones, k, curStone + k) ||
                    backtrack(stones, k + 1, curStone + k + 1)) {
                memo.put(key, true);
                return true;
            }
        }
        memo.put(key, false);
        return false;
    }

    public int divide(int dividend, int divisor) {
        if (dividend == 0) {
            return 0;
        }
        boolean flag = true;
        if ((dividend > 0 && divisor < 0) || (dividend < 0 && divisor > 0)) {
            flag = false;
        }
        if (dividend == Integer.MIN_VALUE) {
            return dividend;
        }
        dividend = Math.abs(dividend);
        divisor = Math.abs(divisor);
        if (flag) {
            return div(dividend, divisor);
        }
        return -div(dividend, divisor);
    }

    int div(int a, int b) {
        if (a < b) {
            return 0;
        }
        int count = 1;
        int tb = b;
        while (tb + tb < a) {
            count += count;
            tb += tb;
        }
        return count + div(a - tb, b);
    }

    public int maxTurbulenceSize(int[] arr) {
        int n = arr.length;
        int[][] dp = new int[n][2];
        int maxLen = 1;
        dp[0][0] = dp[0][1] = 1;//末尾上升为0，下降为1
        for (int i = 0; i < n; i++) {
            dp[i][0] = dp[i][1] = 1;
        }
        for (int j = 1; j < n; j++) {
            if (arr[j] > arr[j - 1]) {
                dp[j][1] = dp[j - 1][0] + 1;
            } else if (arr[j] < arr[j - 1]) {
                dp[j][0] = dp[j - 1][1] + 1;
            }
        }
        for (int i = 0; i < n; i++) {
            maxLen = Math.max(maxLen, Math.max(dp[i][0], dp[i][1]));
        }
        return maxLen;
    }

    public int minSteps(int n) {
        int[] dp = new int[n + 1];
        dp[2] = 2;
        dp[3] = 3;
        for (int i = 4; i <= n; i++) {
            for (int j = i / 2; j >= 1; j--) {
                if (i % j == 0) {
                    int time = i / j;
                    dp[i] = dp[j] + time;
                    break;
                }
            }
        }
        return dp[n];
    }

    public int lastStoneWeightII(int[] stones) {
        PriorityQueue<Integer> q = new PriorityQueue<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return o2 - o1;
            }
        });
        for (int s : stones) {
            q.add(s);
        }
        while (q.size() > 1) {
            int a = q.poll();
            int b = q.poll();
            int nStone = crash(a, b);
            if (nStone != 0) {
                q.offer(nStone);
            }
        }
        if (q.isEmpty()) {
            return 0;
        }
        return q.peek();
    }

    int crash(int a, int b) {
        if (a == b) {
            return 0;
        } else {
            return Math.abs(a - b);
        }
    }

    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        int n = nums2.length;
        int[] ans = new int[n];
        HashMap<Integer, Integer> mem = new HashMap<>();
        Stack<Integer> st = new Stack<>();
        for (int i = n - 1; i >= 0; i--) {
            while (!st.isEmpty() && st.peek() <= nums2[i]) {
                st.pop();
            }
            ans[i] = st.isEmpty() ? -1 : st.peek();
            mem.put(nums2[i], ans[i]);
            st.push(nums2[i]);
        }
        int[] res = new int[nums1.length];
        for (int i = 0; i < nums1.length; i++) {
            res[i] = mem.get(nums1[i]);
        }
        return res;
    }

    public int[] nextGreaterElements(int[] nums) {
        int n = nums.length;
        int[] ans = new int[n];
        Stack<Integer> st = new Stack<>();
        for (int i = 2 * n - 1; i >= 0; i--) {
            while (!st.isEmpty() && nums[st.peek()] <= nums[i % n]) {
                st.pop();
            }
            ans[i % n] = st.isEmpty() ? -1 : i % n;
            st.push(i % n);
        }
        return ans;
    }

    public void solveSudoku(char[][] board) {
        backtrack(board, 0, 0);
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                board[i][j] = h[i][j];
            }
        }
    }

    char[][] h = new char[9][9];

    public void backtrack(char[][] board, int row, int col) {
        if (row == 9) {
            for (int i = 0; i < 9; i++) {
                for (int j = 0; j < 9; j++) {
                    h[i][j] = board[i][j];
                }
            }
            return;
        }
        if (board[row][col] != '.') {
            if (col == 8) {
                backtrack(board, row + 1, 0);
            } else {
                backtrack(board, row, col + 1);
            }
        } else {
            for (char k = '1'; k <= '9'; k++) {
                if (isValid(board, k, row, col)) {
                    board[row][col] = k;
                    if (col == 8) {
                        backtrack(board, row + 1, 0);
                    } else {
                        backtrack(board, row, col + 1);
                    }
                    board[row][col] = '.';
                }
            }
        }
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

    public int maxProfit(int k, int[] prices) {
        int n = prices.length;
        int[][][] dp = new int[n][k + 1][2];
        for (int K = 0; K <= k; K++) {
            dp[0][K][0] = 0;
            dp[0][K][1] = -prices[0];
        }
        for (int i = 1; i < n; i++) {
            for (int K = k; K >= 1; K--) {
                dp[i][K][0] = Math.max(dp[i - 1][K][0], dp[i - 1][K][1] + prices[i]);
                dp[i][K][1] = Math.max(dp[i - 1][K][1], dp[i - 1][K - 1][0] - prices[i]);
            }
        }
        return dp[n - 1][k][0];
    }

    public void nextPermutation(int[] nums) {
        int n = nums.length;
        int first = -1, second = -1;
        for (int i = n - 1; i >= 1; i--) {
            if (nums[i] > nums[i - 1]) {
                first = i - 1;
                second = i;
                break;
            }
        }
        if (first == -1) {
            Arrays.sort(nums);
            return;
        }
        int third = -1;
        for (int i = n - 1; i >= second; i--) {
            if (nums[i] > nums[first]) {
                third = i;
                break;
            }
        }
        if (third != -1) {
            swap(nums, first, third);
        }
        Arrays.sort(nums, first + 1, nums.length);
    }

    void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    public List<Integer> findSubstring(String s, String[] words) {
        HashMap<String, Integer> mp = new HashMap<>();
        int totalLen = words.length * words[0].length();
        int wordLen = words[0].length();
        for (String w : words) {
            mp.put(w, mp.getOrDefault(w, 0) + 1);
        }
        List<Integer> ans = new LinkedList<>();
        if (totalLen > s.length()) {
            return ans;
        }
        int left = 0, right = totalLen;
        while (right <= s.length()) {
            String window = s.substring(left, right);
            HashMap<String, Integer> contrast = new HashMap<>();
            while (window.length() > 0) {
                String key = window.substring(0, wordLen);
                if (mp.containsKey(key) && contrast.getOrDefault(key, 0) < mp.get(key)) {
                    contrast.put(key, contrast.getOrDefault(key, 0) + 1);
                    window = window.substring(wordLen);
                    if (window.equals("")) {
                        ans.add(left);
                    }
                } else {
                    break;
                }
            }
            left++;
            right++;
        }
        return ans;
    }

    public ListNode deleteDuplicates(ListNode head) {
        ListNode p = head;
        if (head == null) {
            return null;
        }
        ListNode nxt = head.next;
        ListNode NewNode = new ListNode(-1);
        NewNode.next = head;
        ListNode pre = NewNode;
        while (nxt != null) {
            if (p.val == nxt.val) {
                while (nxt != null && p.val == nxt.val) {
                    nxt = nxt.next;
                }
                pre.next = nxt;
                p = nxt;
                if (nxt != null) {
                    nxt = nxt.next;
                }
                continue;
            }
            pre = p;
            p = nxt;
            nxt = nxt.next;
        }
        return NewNode.next;
    }

    public String simplifyPath(String path) {
        StringBuilder pa = new StringBuilder(path);
        for (int i = 0; i < pa.length(); i++) {
            if (pa.charAt(i) == '/') {
                while (i + 1 < pa.length() && pa.charAt(i + 1) == '/') {
                    pa.deleteCharAt(i + 1);
                }
            }
        }
        if (pa.charAt(pa.length() - 1) != '/') {
            pa.append('/');
        }
        Stack<String> st = new Stack<>();
        int n = pa.length();
        int start = 0;
        while (start < n) {
            int end = -1;
            for (int i = start + 1; i < n; i++) {
                if (pa.charAt(i) == '/') {
                    end = i;
                    String cur = pa.substring(start + 1, end);
                    if (cur.equals("..")) {
                        if (!st.isEmpty()) {
                            st.pop();
                        }
                    } else if (!cur.equals(".")) {
                        st.push(cur);
                    }
                    start = end;
                    break;
                }
            }
            if (start == n - 1) {
                start++;
            }
        }
        String ans = "/";
        while (!st.isEmpty()) {
            ans = "/" + st.pop() + ans;
        }
        if (ans.equals("/")) {
            return ans;
        }
        return ans.substring(0, ans.length() - 1);
    }

    public List<TreeNode> generateTrees(int n) {
        if (n == 0) {
            return new LinkedList<>();
        }
        return plant(1, n);
    }

    public List<TreeNode> plant(int start, int end) {
        List<TreeNode> forest = new LinkedList<>();
        if (start > end) {
            forest.add(null);
            return forest;
        }
        for (int i = start; i <= end; i++) {
            TreeNode root = new TreeNode(i);
            List<TreeNode> leftFore = plant(start, i - 1);
            List<TreeNode> rightFore = plant(i + 1, end);
            for (TreeNode l : leftFore) {
                for (TreeNode r : rightFore) {
                    root.left = l;
                    root.right = r;
                    forest.add(root);
                }
            }
        }
        return forest;
    }

    public ListNode partition(ListNode head, int x) {
        ListNode pivot = head;
        ListNode nail = new ListNode(-1);
        nail.next = head;
        ListNode q = nail;
        while (pivot != null && pivot.val < x) {
            nail = pivot;
            pivot = pivot.next;
        }
        if (pivot == null) {
            return head;
        }
        ListNode p = pivot.next, pre = pivot;
        while (p != null) {
            if (p.val < x) {
                nail.next = p;
                pre.next = p.next;
                p.next = pivot;
                nail = p;
            }
            pre = p;
            p = p.next;
        }
        return q.next;
    }

    public TreeNode sortedListToBST(ListNode head) {
        return buildTree(head);
    }

    TreeNode buildTree(ListNode head) {
        if (head == null) {
            return null;
        }
        if (head.next == null) {
            return new TreeNode(head.val);
        }
        ListNode slow = head, fast = head, slowPre = null;
        while (fast != null) {
            if (fast.next != null) {
                fast = fast.next.next;
            } else {
                break;
            }
            slowPre = slow;
            slow = slow.next;
        }
        TreeNode root = new TreeNode(slow.val);
        slowPre.next = null;
        root.left = buildTree(head);
        root.right = buildTree(slow.next);
        return root;
    }

    public int maximalRectangle(char[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        int[][] leftOnes = new int[m][n];
        for (int i = 0; i < m; i++) {
            leftOnes[i][0] = (matrix[i][0] == '1' ? 1 : 0);
        }
        for (int i = 0; i < m; i++) {
            for (int j = 1; j < n; j++) {
                leftOnes[i][j] = matrix[i][j] == '1' ? leftOnes[i][j - 1] + 1 : 0;
            }
        }
        int maxArea = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int width = leftOnes[i][j];
                for (int k = i; k >= 0; k--) {
                    if (leftOnes[k][j] == 0) {
                        break;
                    }
                    width = Math.min(width, leftOnes[k][j]);
                    maxArea = Math.max(maxArea, (i - k + 1) * width);
                }
            }
        }
        return maxArea;
    }

    public int largestRectangleArea(int[] heights) {
        int n = heights.length;
        Stack<Integer> st = new Stack<>();
        int maxArea = -1;
        for (int i = 0; i < n; i++) {
            while (!st.isEmpty() && heights[i] < heights[st.peek()]) {
                int depth = 1;
                while (!st.isEmpty() && heights[i] < heights[st.peek()]) {
                    int index = st.pop();
                    maxArea = getMax(maxArea, heights[i] * (i - index + 1), heights[index] * depth);
                    depth++;
                }
            }
            st.push(i);
        }
        while (st.size() > 1) {
            int curbarIndex = st.pop();
            int depth = heights.length - curbarIndex;
            maxArea = Math.max(maxArea, heights[curbarIndex] * depth);
        }
        maxArea = Math.max(maxArea, heights[st.pop()] * heights.length);
        return maxArea;
    }

    int getMax(int a, int b, int c) {
        return Math.max(a, Math.max(b, c));
    }

    public int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> q = new PriorityQueue<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return o2 - o1;
            }
        });
        for (int i : nums) {
            q.add(i);
        }
        for (int i = 0; i < k - 1; i++) {
            q.poll();
        }
        return q.poll();
    }

    public void flatten(TreeNode root) {
        if (root == null) {
            return;
        }
        flatten(root.left);
        flatten(root.right);
        TreeNode left = root.left;
        TreeNode right = root.right;
        root.left = null;
        root.right = left;
        TreeNode p = root;
        while (p.right != null) {
            p = p.right;
        }
        p.right = right;
    }

    public int[] maxSlidingWindow(int[] nums, int k) {
        PriorityQueue<int[]> q = new PriorityQueue<>(new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] != o2[0] ? o2[0] - o1[0] : o2[1] - o1[1];
            }
        });
        int n = nums.length;
        int[] ans = new int[n - k + 1];
        for (int i = 0; i < k; i++) {
            q.offer(new int[]{nums[i], i});
        }
        ans[0] = q.peek()[0];
        for (int i = k; i < n; i++) {
            q.offer(new int[]{nums[i], i});
            while (q.peek()[1] <= i - k) {
                q.poll();
            }
            ans[i - k + 1] = q.peek()[0];
        }
        return ans;
    }

    public List<String> summaryRanges(int[] nums) {
        List<String> ans = new LinkedList<>();
        int n = nums.length;
        int i = 0;
        if (nums.length == 1) {
            ans.add("" + nums[0]);
            return ans;
        }
        if (nums.length == 2) {
            if (nums[1] - nums[0] == 1) {
                ans.add(nums[0] + "->" + nums[1]);
                return ans;
            } else {
                ans.add("" + nums[0]);
                ans.add("" + nums[1]);
                return ans;
            }
        }
        while (i < n) {
            int start = i;
            if (start == n - 1) {
                ans.add("" + nums[start]);
                break;
            }
            boolean isOut = false;
            for (int j = start + 1; j < n; j++) {
                isOut = true;
                if (j == n - 1) {
                    if (nums[j] - nums[j - 1] == 1) {
                        ans.add(nums[start] + "->" + nums[j]);
                    } else {
                        if (start != j - 1) {
                            ans.add(nums[start] + "->" + nums[j - 1]);
                            ans.add("" + nums[j]);
                        } else {
                            ans.add("" + nums[j]);
                        }
                    }
                    i = n;
                    break;
                }
                if (nums[j] - nums[j - 1] > 1) {
                    if (j - 1 == start) {
                        ans.add("" + nums[start]);
                    } else {
                        ans.add(nums[start] + "->" + nums[j - 1]);
                    }
                    i = j;
                    break;
                }
            }
            if (!isOut) {
                break;
            }
        }
        return ans;
    }

    public String minRemoveToMakeValid(String s) {
        int n = s.length();
        Stack<String> st = new Stack<>();
        for (int i = 0; i < n; i++) {
            if (s.charAt(i) == '(') {
                st.push("(" + i);
            } else if (s.charAt(i) == ')') {
                if (!st.isEmpty() && st.peek().charAt(0) == '(') {
                    st.pop();
                } else {
                    st.push(")" + i);
                }
            }
        }
        StringBuilder ans = new StringBuilder(s);
        while (!st.isEmpty()) {
            int index = Integer.parseInt(st.pop().substring(1));
            ans.deleteCharAt(index);
        }
        return ans.toString();
    }

    public int lengthOfLongestSubstring(String s) {
        int n = s.length();
        int left = 0, right = 0;
        int ans = 1;
        LinkedList<Character> window = new LinkedList<>();
        HashMap<Character, Integer> mp = new HashMap<>();
        while (right < n) {
            window.add(s.charAt(right));
            mp.put(s.charAt(right), mp.getOrDefault(s.charAt(right), 0) + 1);
            while (right < n && mp.containsKey(s.charAt(right)) && mp.get(s.charAt(right)) > 1) {
                ans = Math.max(ans, window.size() - 1);
                char l = window.removeFirst();
                mp.put(l, mp.get(l) - 1);
            }
            right++;
        }
        return ans;
    }

    public int maxUncrossedLines(int[] nums1, int[] nums2) {
        int m = nums1.length;
        int n = nums2.length;
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (nums1[i] == nums2[j]) {
                    dp[i + 1][j + 1] = dp[i][j] + 1;
                } else {
                    dp[i + 1][j + 1] = Math.max(dp[i + 1][j], dp[i][j + 1]);
                }
            }
        }
        return dp[m][n];
    }

    public List<List<Integer>> permute(int[] nums) {
        backtrack(nums);
        return Tracks;
    }

    List<List<Integer>> Tracks = new LinkedList<>();
    LinkedList<Integer> Track = new LinkedList<>();

    void backtrack(int[] nums) {
        if (Track.size() == nums.length) {
            Tracks.add(new LinkedList<>(Track));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (!Track.contains(nums[i])) {
                Track.add(nums[i]);
                backtrack(nums);
                Track.removeLast();
            }
        }
    }

    public String MinRemoveToMakeValid(String s) {
        Stack<String> st = new Stack<>();
        int n = s.length();
        for (int i = 0; i < n; i++) {
            if (s.charAt(i) == '(') {
                st.push("(" + i);
            } else if (s.charAt(i) == ')') {
                if (!st.isEmpty() && st.peek().charAt(0) == '(') {
                    st.pop();
                } else {
                    st.push(")" + i);
                }
            }
        }
        StringBuilder sb = new StringBuilder(s);
        while (!st.isEmpty()) {
            int index = Integer.parseInt(st.pop().substring(1));
            sb.deleteCharAt(index);
        }
        return sb.toString();
    }

    public int countBattleships(char[][] board) {
        int m = board.length;
        int n = board[0].length;
        int ans = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] == 'X') {
                    ans++;
                    dfs(board, i, j, m, n);
                }
            }
        }
        return ans;
    }

    int[] dx = {-1, 0, 1, 0};
    int[] dy = {0, 1, 0, -1};

    void dfs(char[][] board, int x, int y, int m, int n) {
        board[x][y] = 'O';
        for (int i = 0; i < 4; i++) {
            int nx = x + dx[i];
            int ny = y + dy[i];
            if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
                if (board[nx][ny] == 'X') {
                    dfs(board, nx, ny, m, n);
                }
            }
        }
    }

    public int maxSubarraySumCircular(int[] nums) {
        int dpMax = nums[0];
        int dpMin = Math.min(0, nums[0]);
        int Max = dpMax;
        int Min = dpMin;
        int sum = nums[0];
        for (int i = 1; i < nums.length; i++) {
            dpMax = Math.max(dpMax + nums[i], nums[i]);
            Max = Math.max(Max, dpMax);
            dpMin = Math.min(dpMin + nums[i], nums[i]);
            Min = Math.min(dpMin, Min);
            sum += nums[i];
        }
        return Math.max(Max, sum - Min);
    }

    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        for (int i = 1; i <= amount; i++) {
            for (int c : coins) {
                if (i - c < 0) {
                    continue;
                }
                dp[i] = dp[i - c] + 1;
            }
        }
        return dp[amount];
    }

    public int maxValue(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int[][] dp = new int[m][n];
        dp[0][0] = grid[0][0];
        for (int i = 1; i < n; i++) {
            dp[0][i] = dp[0][i - 1] + grid[0][i];
        }
        for (int i = 1; i < m; i++) {
            dp[i][0] = dp[i - 1][0] + grid[i][0];
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
            }
        }
        return dp[m - 1][n - 1];
    }

    public int longestValidParentheses(String s) {
        int n = s.length();
        Stack<Character> st = new Stack<>();
        int Max = 0;
        for (int i = 0; i < n; i++) {
            if (s.charAt(i) == '(') {
                st.clear();
                st.push('(');
            }
            for (int j = i + 1; j < n; j++) {
                if (s.charAt(j) == '(') {
                    st.push('(');
                } else {
                    if (st.isEmpty()) {
                        break;
                    } else {
                        st.pop();
                        if (st.isEmpty()) {
                            Max = Math.max(Max, j - i + 1);
                        }
                    }
                }
            }
        }
        return Max;
    }

    public List<String> restoreIpAddresses(String s) {
        backtrack(s, 0, 0);
        return tracs;
    }

    StringBuilder trac = new StringBuilder();
    List<String> tracs = new LinkedList<>();

    void backtrack(String s, int index, int segs) {
        if (index == s.length() && segs == 4) {
            tracs.add(trac.substring(0, trac.length() - 1));
            return;
        }
        for (int i = index; i < s.length(); i++) {
            if (s.charAt(index) == '0' && i > index) {
                break;
            }
            int pivot = Integer.parseInt(s.substring(index, i + 1));
            if (pivot > 255) {
                break;
            }
            if (pivot >= 0 && pivot <= 255) {
                trac.append(s, index, i + 1).append('.');
                backtrack(s, i + 1, segs + 1);
                trac.deleteCharAt(trac.length() - 1);
                trac.delete(trac.length() - i - 1 + index, trac.length());
            }
        }
    }

    public void recoverTree(TreeNode root) {
        List<Integer> nums = new LinkedList<>();
        inorder(nums, root);
        int[] arr = findCandidate(nums);
        recover(arr[0], arr[1], root, 2);
    }

    void inorder(List<Integer> nums, TreeNode root) {
        if (root == null) {
            return;
        }
        inorder(nums, root.left);
        nums.add(root.val);
        inorder(nums, root.right);
    }

    int[] findCandidate(List<Integer> nums) {
        int n = nums.size();
        int x = -1, y = -1;
        for (int i = 0; i < n - 1; ++i) {
            if (nums.get(i + 1) < nums.get(i)) {
                y = nums.get(i + 1);
                if (x == -1) {
                    x = nums.get(i);
                } else {
                    break;
                }
            }
        }
        return new int[]{x, y};
    }

    void recover(int x, int y, TreeNode root, int count) {
        if (root == null) {
            return;
        }
        if (root.val == x || root.val == y) {
            root.val = root.val == x ? y : x;
            count--;
            if (count == 0) {
                return;
            }
        }
        recover(x, y, root.left, count);
        recover(x, y, root.right, count);

    }

    public List<String> letterCombinations(String digits) {
        HashMap<Character, String> mp = new HashMap<>();
        mp.put('2', "abc");
        mp.put('3', "def");
        mp.put('4', "ghi");
        mp.put('5', "jkl");
        mp.put('6', "mno");
        mp.put('7', "pqrs");
        mp.put('8', "tuv");
        mp.put('9', "wxyz");
        backtrack(digits, mp, 0);
        return combs;
    }

    StringBuilder comb = new StringBuilder();
    List<String> combs = new LinkedList<>();

    void backtrack(String digits, HashMap<Character, String> mp, int index) {
        if (index == digits.length() && comb.length() == digits.length()) {
            combs.add(comb.toString());
            return;
        }
        String candidate = mp.get(digits.charAt(index));
        for (int j = 0; j < candidate.length(); j++) {
            comb.append(candidate.charAt(j));
            backtrack(digits, mp, index + 1);
            comb.deleteCharAt(comb.length() - 1);
        }
    }

    public int maxArea(int[] height) {
        int n = height.length;
        int left = 0, right = n - 1;
        int area = 0;
        while (left < right) {
            area = Math.max((right - left) * Math.min(height[left], height[right]), area);
            if (height[left] < height[right]) {
                left++;
            } else {
                right--;
            }
        }
        return area;
    }
    public String longestCommonPrefix(String[] strs) {
        StringBuilder ans= new StringBuilder();
        int n=strs.length;
        int len=strs[0].length();
        for(int i=0;i<len;i++){
            char cur=strs[0].charAt(i);
            for(int j=1;j<n;j++){
                if(strs[j].length()-1<i||strs[j].charAt(i)!=cur){
                    return ans.toString();
                }
            }
            ans.append(cur);
        }
        return ans.toString();
    }
    public int jump(int[] nums) {
        int n=nums.length;
        int furthest=0;
        int step=0;
        for(int i=0;i<n;i++){
            if(i>furthest){
                return -1;
            }
            for(int j=i;j<=i+nums[i];j++){
                furthest=Math.max(furthest,j);
                if(furthest>=n-1){
                    return step+1;
                }
            }
            step++;
        }
        return -1;
    }
    public static void main(String[] args) {
        Practice pr = new Practice();
        int[] nums1 = {1, 4, 2}, nums2 = {1, 2, 4};
        System.out.println(pr.maxUncrossedLines(nums1, nums2));
        String s = "a)b(c)d";
        System.out.println(pr.MinRemoveToMakeValid(s));
        int[] coins = {1, 2, 5};
        int amount = 11;
        System.out.println(pr.coinChange(coins, amount));
        String ss = ")()())";
        System.out.println(pr.longestValidParentheses(ss));
        s = "25525511135";
        System.out.println(pr.restoreIpAddresses(s));
        Integer[] root = {1, 3, null, null, 2};
        TreeNode r = ConstructTree.constructTree(root);
        pr.recoverTree(r);
        TreeOperations.show(r);
        String digits = "23";
        System.out.println(pr.letterCombinations(digits));
        int[] heights = {1, 8, 6, 2, 5, 4, 8, 3, 7};
        System.out.println(pr.maxArea(heights));
        String[]strs = {"flower","flow","flight"};
        System.out.println(pr.longestCommonPrefix(strs));
        int[]nums = {2,3,1,1,4};
        System.out.println(pr.jump(nums));
    }
}
