import java.util.*;
import java.lang.*;

public class drill {
    public ListNode oddEvenList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode p = head, q = head.next, memo = q, Ppre = null;
        while (p != null && q != null) {
            p.next = q.next;
            if (q.next != null) {
                q.next = q.next.next;
            }
            Ppre = p;
            p = p.next;
            q = q.next;
        }
        if (p != null) {
            p.next = memo;
        } else {
            Ppre.next = memo;
        }
        return head;
    }

    int[][] Matrix;
    int[] dx = {-1, 0, 1, 0};
    int[] dy = {0, 1, 0, -1};
    int ans = 0;
    boolean[][] visited;
    int[][] memo;

    public int longestIncreasingPath(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        Matrix = matrix;
        visited = new boolean[m][n];
        memo = new int[m][n];
        int res = 1;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                res = Math.max(backtrack(i, j), res);
                trackLen = 1;
                for (int k = 0; k < m; k++) {
                    Arrays.fill(visited[k], false);
                }
            }
        }
        return res;
    }

    int trackLen = 1;

    public int backtrack(int x, int y) {
        if (memo[x][y] != 0) {
            return memo[x][y];
        }
        if (isEnd(x, y)) {
            ans = Math.max(ans, trackLen);
            memo[x][y] = ans;
            return memo[x][y];
        }
        visited[x][y] = true;
        int m = Matrix.length, n = Matrix[0].length;
        for (int i = 0; i < 4; i++) {
            int nx = x + dx[i];
            int ny = y + dy[i];
            if (nx >= 0 && nx < m && ny >= 0 && ny < n && !visited[nx][ny] && Matrix[nx][ny] > Matrix[x][y]) {
                trackLen++;
                backtrack(nx, ny);
                trackLen--;
            }
        }
        visited[x][y] = false;
        memo[x][y] = ans;
        return memo[x][y];
    }

    boolean isEnd(int x, int y) {
        int m = Matrix.length, n = Matrix[0].length;
        for (int i = 0; i < 4; i++) {
            int nx = x + dx[i];
            int ny = y + dy[i];
            if (nx >= 0 && nx < m && ny >= 0 && ny < n && !visited[nx][ny] && Matrix[nx][ny] > Matrix[x][y]) {
                return false;
            }
        }
        return true;
    }

    public boolean increasingTriplet(int[] nums) {
        int small = Integer.MAX_VALUE, mid = Integer.MAX_VALUE;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] < small) {
                small = nums[i];
            } else if (nums[i] < mid) {
                mid = nums[i];
            } else if (nums[i] > mid) {
                return true;
            }
        }
        return false;
    }

    public int[] topKFrequent(int[] nums, int k) {
        HashMap<Integer, Integer> mp = new HashMap<>();
        for (int n : nums) {
            mp.put(n, mp.getOrDefault(n, 0) + 1);
        }
        PriorityQueue<Integer> q = new PriorityQueue<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return mp.get(o2) - mp.get(o1);
            }
        });
        q.addAll(mp.keySet());
        int[] ans = new int[k];
        for (int i = 0; i < k; i++) {
            ans[i] = q.poll();
        }
        return ans;
    }

    public int[] mergeSort(int[] nums, int left, int right) {
        if (right == left) {
            return new int[]{nums[left]};
        }
        int mid = left + (right - left) / 2;
        int[] leftArr = mergeSort(nums, left, mid);
        int[] rightArr = mergeSort(nums, mid + 1, right);
        int[] result = new int[leftArr.length + rightArr.length];
        int cur = 0, p = 0, q = 0;
        while (p < leftArr.length && q < rightArr.length) {
            result[cur++] = leftArr[p] < rightArr[q] ? leftArr[p++] : rightArr[q++];
        }
        while (p < leftArr.length) {
            result[cur++] = leftArr[p++];
        }
        while (q < rightArr.length) {
            result[cur++] = rightArr[q++];
        }
        return result;
    }


    List<String> res = new LinkedList<>();
    String track = "";


//
//    public int reversePairs(int[] nums) {
//        int[] result = mergeCount(nums, 0, nums.length - 1);
//        System.out.println(Arrays.toString(result));
//        return k;
//    }
//
//    int k;
//
//    public int[] mergeCount(int[] nums, int left, int right) {
//        if (right == left) {
//            return new int[]{nums[left]};
//        }
//        int mid = left + (right - left) / 2;
//        int[] leftArr = mergeCount(nums, left, mid);
//        int[] rightArr = mergeCount(nums, mid + 1, right);
//
//        int[] result = new int[leftArr.length + rightArr.length];
//        int cur = 0, p = 0, q = 0;
//        while (p < leftArr.length && q < rightArr.length) {
//            if (leftArr[p] > rightArr[q]) {
//                result[cur++] = leftArr[p++];
//                k += rightArr.length - q;
//            } else {
//                result[cur++] = rightArr[q++];
//            }
//        }
//        while (p < leftArr.length) {
//            result[cur++] = leftArr[p++];
//        }
//        while (q < rightArr.length) {
//            result[cur++] = rightArr[q++];
//        }
//        return result;
//    }

    public int networkDelayTime(int[][] times, int n, int k) {
        HashMap<Integer, List<int[]>> map = new HashMap<>();
        for (int[] time : times) {
            if (!map.containsKey(time[0])) {
                map.put(time[0], new LinkedList<>());
            }
            map.get(time[0]).add(new int[]{time[1], time[2]});
        }
        HashMap<Integer, Integer> dist = new HashMap<>();
        for (int i = 1; i <= n; i++) {
            dist.put(i, Integer.MAX_VALUE);
        }
        boolean[] visited = new boolean[n + 1];
        dist.put(k, 0);
        int visit = 0;
        while (true) {
            int candidate = -1;
            int candDist = Integer.MAX_VALUE;
            for (int i = 1; i <= n; i++) {
                if (!visited[i] && dist.get(i) < candDist) {
                    candidate = i;
                    candDist = dist.get(i);
                }
            }
            if (candidate == -1) {
                break;
            }
            visited[candidate] = true;
            visit++;
            dist.put(candidate, candDist);
            if (map.containsKey(candidate)) {
                for (int[] item : map.get(candidate)) {
                    int dis = Math.min(dist.get(item[0]), candDist + item[1]);
                    dist.put(item[0], dis);
                }
            }
        }
        int ans = -1;
        for (int i : dist.keySet()) {
            if (dist.get(i) != Integer.MAX_VALUE) {
                ans = Math.max(ans, dist.get(i));
            }
        }
        if (visit != n) {
            return -1;
        }
        return ans == 0 ? -1 : ans;
    }

    public int findCheapestPrice(int n, int[][] flights, int src, int dst, int K) {
        int[][] graph = new int[n][n];
        for (int[] f : flights) {
            graph[f[0]][f[1]] = f[2];
        }
        PriorityQueue<int[]> q = new PriorityQueue<>(new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[2] - o2[2];
            }
        });
        q.offer(new int[]{src, K + 1, 0});
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int v = cur[0], k = cur[1], cost = cur[2];
            if (v == dst) {
                return cost;
            }
            if (k > 0) {
                for (int i = 0; i < n; i++) {
                    if (graph[v][i] != 0) {
                        q.offer(new int[]{i, k - 1, cost + graph[v][i]});
                    }
                }
            }
        }
        return -1;
    }

    public int[] warmerDay(int[] degrees) {
        int n = degrees.length;
        int[] ans = new int[n];
        Stack<Integer> st = new Stack<>();
        for (int i = n - 1; i >= 0; i--) {
            while (!st.isEmpty() && degrees[i] >= degrees[st.peek()]) {
                st.pop();
            }
            ans[i] = st.isEmpty() ? 0 : st.peek() - i;
            st.push(i);
        }
        return ans;
    }

    public int cuttingRope(int n) {
        int[] dp = new int[n + 1];
        dp[1] = 1;
        dp[2] = 2;
        dp[3] = 3;
        for (int i = 4; i <= n; i++) {
            for (int j = 1; j < i; j++) {
                dp[i] = Math.max(dp[i], dp[j] * dp[i - j]);
            }
        }
        return dp[n];
    }

    public int[] spiralOrder(int[][] matrix) {
        int m = matrix.length;
        if (m == 0) {
            return new int[]{};
        }
        int n = matrix[0].length;
        if (n == 0) {
            return new int[]{};
        }
        int[][] directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        int way = 0;
        int x = 0, y = 0;
        int cur = 0;
        int[] ans = new int[m * n];
        boolean[][] visited = new boolean[m][n];
        while (cur < m * n) {
            ans[cur++] = matrix[x][y];
            visited[x][y] = true;
            if (!isBlocked(x, y, directions[way], m, n, visited)) {
                x += directions[way][0];
                y += directions[way][1];
            } else {
                way = (way + 1) % 4;
                x += directions[way][0];
                y += directions[way][1];
            }
        }
        return ans;
    }


    public int numBusesToDestination(int[][] routes, int source, int target) {
        if (source == target) {
            return 0;
        }
        HashMap<Integer, List<Integer>> G = new HashMap<>();
        HashSet<Integer> visited = new HashSet<>();
        for (int i = 0; i < routes.length; i++) {
            Arrays.sort(routes[i]);
            G.put(i, new LinkedList<>());
        }

        for (int i = 0; i < routes.length; i++) {
            for (int j = i + 1; j < routes.length; j++) {
                if (isCrossed(routes[i], routes[j])) {
                    G.get(i).add(j);
                    G.get(j).add(i);
                }
            }
        }
        Queue<Integer> q = new LinkedList<>();
        HashSet<Integer> S = new HashSet<>();
        HashSet<Integer> T = new HashSet<>();
        for (int i = 0; i < routes.length; i++) {
            if (Arrays.binarySearch(routes[i], source) >= 0) {
                S.add(i);
            }
            if (Arrays.binarySearch(routes[i], target) >= 0) {
                T.add(i);
            }
        }
        q.addAll(S);
        visited.addAll(S);
        int step = 1;
        while (!q.isEmpty()) {
            int sz = q.size();
            for (int i = 0; i < sz; i++) {
                int cur = q.poll();
                if (T.contains(cur)) {
                    return step;
                }
                for (int bus : G.get(cur)) {
                    if (visited.contains(bus)) {
                        continue;
                    }
                    q.offer(bus);
                    visited.add(bus);
                }
            }
            step++;
        }
        return -1;
    }

    public boolean isCrossed(int[] A, int[] B) {
        int i = 0, j = 0;
        while (i < A.length && j < B.length) {
            if (A[i] == B[j]) {
                return true;
            }
            if (A[i] < B[j]) {
                i++;
            } else {
                j++;
            }
        }
        return false;
    }

    public int findNthDigit(int n) {
        long start = 1, count = 9;
        int digit = 1;
        while (n > count) {
            n -= count;
            start *= 10;
            digit++;
            count = 9 * start * digit;
        }
        long num = start + (n - 1) / digit;
//        int bit=(n-1)%digit;
        return Long.toString(num).charAt((n - 1) % digit) - '0';
    }

    public int countDigitOne(int n) {
        int low = 0, cur = 0, high = n;
        int count = 0;
        int num = 1;
        while (high > 0) {
            cur = high % 10;
            high /= 10;
            if (cur == 0) {
                count += high * num;
            } else if (cur == 1) {
                count += high * num + low + 1;
            } else if (cur > 1) {
                count += (high + 1) * num;
            }
//            low=high%num;
            num *= 10;
            low = n % num;
        }
        return count;
    }


    public void sortColors(int[] nums) {
        int p0 = 0, p1 = 0;
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            if (nums[i] == 1) {
                swap(nums, p1, i);
                p1++;
            } else if (nums[i] == 0) {
                swap(nums, p0, i);
                if (p0 < p1) {
                    swap(nums, p1, i);
                }
                p1++;
                p0++;
            }
        }
    }

    public int myMax(int a, int b, int c) {
        return Math.max(a, Math.max(b, c));
    }

    public int largestRectangleArea(int[] heights) {
        int n = heights.length;
        Stack<int[]> st = new Stack<>();
        int area = 0;
        for (int i = 0; i < n; i++) {
            while (!st.isEmpty() && heights[i] < st.peek()[0]) {
                int[] cur = st.pop();
                int pivot;
                if (!st.isEmpty()) {
                    pivot = i - 1 - st.peek()[1];
                } else {
                    pivot = i;
                }
                area = myMax(area, (i - cur[1] + 1) * heights[i], pivot * cur[0]);
            }
            st.push(new int[]{heights[i], i});
        }
        while (!st.isEmpty()) {
            int[] cur = st.pop();
            int height = cur[0], index = cur[1];
            int dist = n;
            if (!st.isEmpty()) {
                dist = n - st.peek()[1] - 1;
            }
            area = Math.max(area, dist * height);
        }
        return area;
    }

    public int[][] generateMatrix(int n) {
        if (n == 0) {
            return new int[][]{{}};
        }
        int[][] ans = new int[n][n];
        int[][] directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        int way = 0;
        boolean[][] visited = new boolean[n][n];
        int curX = 0, curY = 0;
        int curNum = 1;
        while (!isBlocked(curX, curY, directions[way], n, n, visited)) {
            ans[curX][curY] = curNum;
            visited[curX][curY] = true;
            curNum++;
            curX = curX + directions[way][0];
            curY = curY + directions[way][1];
            if (isBlocked(curX, curY, directions[way], n, n, visited)) {
                way = (way + 1) % 4;
            }
        }
        ans[curX][curY] = curNum;
        return ans;
    }

    public boolean isBlocked(int x, int y, int[] direction, int m, int n, boolean[][] visited) {
        int nx = x + direction[0], ny = y + direction[1];
        if (nx < 0 || nx >= m || ny < 0 || ny >= n || visited[nx][ny]) {
            return true;
        }
        return false;
    }

    int maxAns = Integer.MIN_VALUE;

    int thmax(int a, int b, int c) {
        return Math.max(a, Math.max(b, c));
    }

    public int maxPathSum(TreeNode root) {
        if (root == null) {
            return 0;
        }
        maxGain(root);
        return maxAns;
    }

    int maxGain(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int leftGain = Math.max(maxGain(root.left), 0);
        int rightGain = Math.max(maxGain(root.right), 0);
        int maxPrice = leftGain + rightGain + root.val;
        maxAns = Math.max(maxAns, maxPrice);
        return root.val + Math.max(leftGain, rightGain);
    }

    public int[] countBits(int num) {
        int[] dp = new int[num + 1];
        dp[0] = 0;
        dp[1] = 1;
        for (int i = 2; i <= num; i++) {
            if (i % 2 == 0) {
                dp[i] = dp[i / 2];
            } else {
                dp[i] = dp[i - 1] + 1;
            }
        }
        return dp;
    }

    public String decodeString(String s) {
        int n = s.length();
        int num = 0;
        String res = "";
        Stack<Integer> num_st = new Stack<>();
        Stack<String> str_st = new Stack<>();
        for (int i = 0; i < n; i++) {
            char c = s.charAt(i);
            if (c >= '0' && c <= '9') {
                num = num * 10 + c - '0';
            } else if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
                res = res + c;
            } else if (c == '[') {
                num_st.push(num);
                str_st.push(res);
                num = 0;
                res = "";
            } else if (c == ']') {
                int times = num_st.pop();
                String cur = str_st.pop();
                for (int j = 0; j < times; j++) {
                    cur = cur + res;
                }
                res = cur;
            }
        }
        return res;
    }

    public int pathSum(TreeNode root, int targetSum) {
        backtrack(root, targetSum, 0);
        return trans;
    }

    LinkedList<Integer> trac = new LinkedList<>();
    int trans = 0;

    public void backtrack(TreeNode root, int target, int curSum) {
        if (root == null) {
            return;
        }
        if (curSum == target) {
            trans++;
        }
        trac.add(root.val);
        curSum += root.val;
        backtrack(root.left, target, curSum);
        backtrack(root.right, target, curSum);
        trac.removeLast();
        curSum -= root.val;
        backtrack(root.left, target, curSum);
        backtrack(root.right, target, curSum);
    }

    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode p = l1, q = l2;
        ListNode head = new ListNode(-1);
        ListNode pre = head;
        while (p != null && q != null) {
            if (p.val < q.val) {
                ListNode cur = new ListNode(p.val);
                pre.next = cur;
                pre = cur;
                p = p.next;
            } else {
                ListNode cur = new ListNode(q.val);
                pre.next = cur;
                pre = cur;
                q = q.next;
            }
        }
        while (p != null) {
            ListNode cur = new ListNode(p.val);
            pre.next = cur;
            pre = cur;
            p = p.next;
        }
        while (q != null) {
            ListNode cur = new ListNode(q.val);
            pre.next = cur;
            pre = cur;
            q = q.next;
        }
        return head.next;
    }

    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> ans = new LinkedList<>();
        if (root == null) {
            return ans;
        }
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);

        while (!q.isEmpty()) {
            int sz = q.size();
            List<Integer> level = new LinkedList<>();
            for (int i = 0; i < sz; i++) {
                TreeNode cur = q.poll();
                level.add(cur.val);
                if (cur.left != null) {
                    q.offer(cur.left);
                }
                if (cur.right != null) {
                    q.offer(cur.right);
                }
            }
            ans.add(level);
        }
        return ans;
    }

    public int singleNumber(int[] nums) {
        int n = nums.length;
        int[] res = new int[32];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < 32; j++) {
                res[j] += nums[i] & 1;
                nums[i] >>>= 1;
            }
        }
        for (int i = 0; i < 32; i++) {
            res[i] %= 3;
        }
        int ans = 0;
        for (int i = 31; i >= 0; i--) {
            if (res[i] != 0) {
                ans += Math.pow(2, i);
            }
        }
        return ans;
    }

    public int nthUglyNumber(int n) {
        int[] dp = new int[n];
        int p2 = 0, p3 = 0, p5 = 0;
        dp[0] = 1;
        for (int i = 1; i < n; i++) {
            int a = dp[p2] * 2, b = dp[p3] * 3, c = dp[p5] * 5;
            dp[i] = Math.min(a, Math.min(b, c));
            if (dp[i] == a) {
                p2++;
            }
            if (dp[i] == b) {
                p3++;
            }
            if (dp[i] == c) {
                p5++;
            }
        }
        return dp[n - 1];
    }

    public int lastRemaining(int n, int m) {
        int x = 0;
        for (int i = 2; i <= n; i++) {
            x = (x + m) % i;
        }
        return x;
    }

    public boolean validateStackSequences(int[] pushed, int[] popped) {
        int n = pushed.length;
        int cur = 0;
        Stack<Integer> st = new Stack<>();
        for (int i = 0; i < n; i++) {
            st.push(pushed[i]);
            while (!st.empty() && st.peek() == popped[cur]) {
                st.pop();
                cur++;
            }
        }
        return st.isEmpty();
    }

    class Node {
        public int val;
        public Node left;
        public Node right;

        public Node() {
        }

        public Node(int _val) {
            val = _val;
        }

        public Node(int _val, Node _left, Node _right) {
            val = _val;
            left = _left;
            right = _right;
        }
    }

    Node pre, head;

    public Node treeToDoublyList(Node root) {
        if (root == null) {
            return null;
        }
        dfs(root);
        pre.left = head;
        head.right = pre;
        return head;
    }

    void dfs(Node cur) {
        if (cur == null) {
            return;
        }
        dfs(cur.left);
        if (pre != null) {
            pre.right = cur;
        } else {
            head = cur;
        }
        cur.left = pre;
        pre = cur;
        dfs(cur.right);
    }

    public String convertToTitle(int columnNumber) {
        int n = columnNumber;
        String ans = "";
        while (n > 0) {
            int cur = n % 26;
            char bit = (char) (cur - 1 + 'A');
            if (cur == 0) {
                bit = 'Z';
            }
            ans = bit + ans;
            if (n == 26) {
                break;
            }
            n /= 26;
        }
        return ans;
    }

    HashMap<String, Boolean> mem = new HashMap<>();

    public boolean checkValidString(String s) {
        return backtrack(s, 0, 0, 0);
    }

    boolean backtrack(String s, int left, int right, int cur) {
        String key = left + "," + right + "," + cur;
        if (mem.containsKey(key)) {
            return mem.get(key);
        }
        if (cur == s.length()) {
            return left == right;
        }
        if (right > left) {
            return false;
        }
        if (s.charAt(cur) == '(') {
            if (backtrack(s, left + 1, right, cur + 1)) {
                mem.put(key, true);
                return true;
            }
        } else if (s.charAt(cur) == ')') {
            if (backtrack(s, left, right + 1, cur + 1)) {
                mem.put(key, true);
                return true;
            }
        } else {
            if (backtrack(s, left + 1, right, cur + 1) || backtrack(s, left, right + 1, cur + 1) ||
                    backtrack(s, left, right, cur + 1)) {
                mem.put(key, true);
                return true;
            }
        }
        mem.put(key, false);
        return false;
    }


    void quick_sort(int[] arr, int left, int right) {
        if (left < right) {
            int pivot = partition(arr, left, right);
            quick_sort(arr, left, pivot - 1);
            quick_sort(arr, pivot + 1, right);
        }
    }

    public int kthFactor(int n, int k) {
        if (k == 1) {
            return 1;
        }
        boolean[] table = new boolean[n + 1];
        table[1] = true;
        int seq = 1;
        int cur = 2;
        while (cur <= n / 2) {
            if (n % cur == 0) {
                table[cur] = true;
                seq++;
                if (seq == k) {
                    return cur;
                }
            }
            cur++;
        }
        if (seq == k - 1) {
            return n;
        }
        return -1;
    }

    public int minMeetingRooms(int[][] intervals) {
        PriorityQueue<int[]> q = new PriorityQueue<>(new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                if (o1[0] == o2[0]) {
                    return o1[1] - o2[1];
                } else {
                    return o1[0] - o2[0];
                }
            }
        });
        for (int[] interv : intervals) {
            q.offer(new int[]{interv[0], 1});
            q.offer(new int[]{interv[1], -1});
        }
        int res = 0;
        int ans = 0;
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            res += cur[1];
            ans = Math.max(ans, res);
        }
        return ans;
    }

    public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        traverse(root, p);
        return an;
    }

    TreeNode Pre = null;
    TreeNode an = null;

    public void traverse(TreeNode root, TreeNode p) {
        if (root == null) {
            return;
        }
        traverse(root.right, p);
        if (root.val > p.val) {
            Pre = root;
        } else {
            an = Pre;
            return;
        }
        traverse(root.left, p);
    }

    public int movingCount(int m, int n, int k) {
        boolean[][] visited = new boolean[m][n];
        return dfs(0, 0, k, m, n, visited);
    }

    int[] di = {-1, 0, 1, 0}, dj = {0, -1, 0, 1};

    public int dfs(int x, int y, int k, int m, int n, boolean[][] visited) {
        if (sumbit(x) + sumbit(y) <= k) {
            ans += 1;
            visited[x][y] = true;
        }
        for (int i = 0; i < 4; i++) {
            int ni = x + di[i];
            int nj = y + dj[i];
            if (ni >= 0 && ni < m && nj >= 0 && nj < n && !visited[ni][nj] && sumbit(x) + sumbit(y) <= k) {
                dfs(ni, nj, k, m, n, visited);
            }
        }
        return ans;
    }

    int sumbit(int num) {
        int ans = 0;
        while (num > 0) {
            ans += num % 10;
            num /= 10;
        }
        return ans;
    }

    public int[] singleNumbers(int[] nums) {
        int x = 0, y = 0, z = 0;
        for (int n : nums) {
            z ^= n;
        }
        int m = 1;
        while ((z & m) == 0) {
            m <<= 1;
        }
        for (int n : nums) {
            if ((n & m) == 0) {
                x ^= n;
            } else {
                y ^= n;
            }
        }
        return new int[]{x, y};
    }

    int[] temp;
    int[] tempIndex;
    int[] index;
    int[] Ans;

    public List<Integer> countSmaller(int[] nums) {
        int n = nums.length;
        temp = new int[n];
        tempIndex = new int[n];
        index = new int[n];
        Ans = new int[n];
        for (int i = 0; i < n; i++) {
            index[i] = i;
        }
        mergesort(nums, 0, n - 1);
        List<Integer> ans = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            ans.add(Ans[i]);
        }
        return ans;
    }

    public void mergesort(int[] nums, int left, int right) {
        if (left >= right) {
            return;
        }
        int mid = (left + right) / 2;
        mergesort(nums, left, mid);
        mergesort(nums, mid + 1, right);
        merge(nums, left, mid, right);
    }

    public void merge(int[] nums, int left, int mid, int right) {
        int i = left, j = mid + 1, cur = left;
        while (i <= mid && j <= right) {
            if (nums[i] > nums[j]) {
                temp[cur] = nums[i];
                tempIndex[cur] = index[i];
                Ans[tempIndex[cur]] += right - j + 1;
                i++;
                cur++;
            } else {
                temp[cur] = nums[j];
                tempIndex[cur] = index[j];
                j++;
                cur++;
            }
        }
        while (i <= mid) {
            temp[cur] = nums[i];
            tempIndex[cur] = index[i];
//            Ans[tempIndex[cur]]+=(j-mid-1);
            i++;
            cur++;
        }
        while (j <= right) {
            temp[cur] = nums[j];
            tempIndex[cur] = index[j];
            j++;
            cur++;
        }
        for (int k = left; k <= right; k++) {
            nums[k] = temp[k];
            index[k] = tempIndex[k];
        }
    }

    HashMap<Integer, List<Integer>> mp2 = new HashMap<>();

    public int maxUncrossedLines(int[] nums1, int[] nums2) {
        for (int i = 0; i < nums2.length; i++) {
            if (!mp2.containsKey(nums2[i])) {
                mp2.put(nums2[i], new LinkedList<>());
                mp2.get(nums2[i]).add(i);
            } else {
                mp2.get(nums2[i]).add(i);
            }
        }
        return backtrack(nums1, 0, -1);
    }

    HashMap<String, Integer> mm = new HashMap<>();

    int backtrack(int[] nums1, int cur, int pre) {
        String key = cur + "," + pre;
        int res = 0;
        if (mm.containsKey(key)) {
            return mm.get(key);
        }
        if (cur == nums1.length) {
            return 0;
        }
        if (!mp2.containsKey(nums1[cur])) {
            res = backtrack(nums1, cur + 1, pre);
        } else {
            res = backtrack(nums1, cur + 1, pre);
            for (int n : mp2.get(nums1[cur])) {
                if (n <= pre) {
                    continue;
                }
                int res1 = 1 + backtrack(nums1, cur + 1, n);
                res = Math.max(res, res1);
            }
        }
        mm.put(key, res);
        return mm.get(key);
    }

    int getmax(int a, int b, int c) {
        return Math.max(a, Math.max(b, c));
    }

    public boolean Find(int target, int[][] array) {
        int m = array.length;
        if (m == 0) {
            return false;
        }
        int n = array[0].length;
        if (n == 0) {
            return false;
        }
        int up = 0, down = m;
        while (up < down) {
            int mid = up + (down - up) / 2;
            if (findLine(array[mid], target)) {
                return true;
            } else {
                if (target < array[mid][0]) {
                    down = mid;
                } else if (target > array[mid][n - 1]) {
                    up = mid + 1;
                } else {
                    if (findLine(array[up], target)) {
                        return true;
                    } else {
                        up++;
                    }
                }
            }
        }
        return false;
    }

    boolean findLine(int[] nums, int target) {
        int n = nums.length;
        int left = 0, right = n;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return true;
            } else if (target > nums[mid]) {
                left = mid + 1;
            } else if (target < nums[mid]) {
                right = mid;
            }
        }
        return false;
    }

    public long maxWater(int[] arr) {
        int n = arr.length;
        int left = 0, right = n - 1;
        int ans = 0;
        int leftmax = arr[0];
        int rightmax = arr[n - 1];
        while (left <= right) {
            leftmax = Math.max(leftmax, arr[left]);
            rightmax = Math.max(rightmax, arr[right]);
            if (leftmax < rightmax) {
                ans += leftmax - arr[left];
                left++;
            } else {
                ans += rightmax - arr[right];
                right--;
            }
        }
        return ans;
    }

    public int findRepeatNumber(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            if (nums[i] == i) {
                continue;
            } else if (nums[nums[i]] == nums[i]) {
                return nums[i];
            } else {
                swap(nums, i, nums[i]);
            }
        }
        return -1;
    }

    void swap(int[] nums, int x, int y) {
        int temp = nums[x];
        nums[x] = nums[y];
        nums[y] = temp;
    }

    public int[] getLeastNumbers(int[] arr, int k) {
        if (k >= arr.length) {
            return arr;
        }
        quicksort(arr, 0, arr.length - 1, k);
        return ansArr;
    }

    int partition(int[] arr, int left, int right) {
        int pivot = left;
        int index = pivot + 1;
        for (int i = index; i <= right; i++) {
            if (arr[i] < arr[pivot]) {
                swap(arr, i, pivot);
            }
        }
        swap(arr, pivot, index - 1);
        return index - 1;
    }

    int[] ansArr;

    void quicksort(int[] arr, int left, int right, int k) {
        if (left <= right) {
            int pivot = partition(arr, left, right);
            if (pivot == k) {
                ansArr = Arrays.copyOfRange(arr, 0, k);
            } else if (pivot < k) {
                quicksort(arr, pivot + 1, right, k);
            } else {
                quicksort(arr, left, pivot - 1, k);
            }
        }
    }

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        if (preorder.length == 0 || inorder.length == 0) {
            return null;
        }
        int root = preorder[0];
        int pivot = -1;
        for (int i = 0; i < inorder.length; i++) {
            if (inorder[i] == root) {
                pivot = i;
                break;
            }
        }
        int leftNum = pivot;
        int rightNum = inorder.length - leftNum - 1;
        TreeNode Root = new TreeNode(root);
        Root.left = buildTree(Arrays.copyOfRange(preorder, 1, leftNum + 1),
                Arrays.copyOfRange(inorder, 0, pivot));
        Root.right = buildTree(Arrays.copyOfRange(preorder, leftNum + 1, preorder.length),
                Arrays.copyOfRange(inorder, pivot + 1, inorder.length));
        return Root;
    }

    public void solveSudoku(char[][] board) {
        ansboard = new char[board.length][board[0].length];
        backtrack(board, 0, 0);
    }

    char[][] ansboard;

    public void backtrack(char[][] board, int row, int col) {
        int m = board.length;
        int n = board[0].length;
        if (row == m) {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    ansboard[i][j] = board[i][j];
                }
            }
            return;
        }
        if (col == n) {
            backtrack(board, row + 1, 0);
        }
        if (board[row][col] != '.') {
            if (col == n - 1) {
                backtrack(board, row + 1, 0);
            } else {
                backtrack(board, row, col + 1);
            }
        } else {
            for (char c = '1'; c <= '9'; c++) {
                if (isValid(board, c, row, col)) {
                    board[row][col] = c;
                    if (col == n - 1) {
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

    public int reversePairs(int[] nums) {
        int[] res = mergeCount(nums, 0, nums.length - 1);
        return N;
    }

    int N;

    int[] mergeCount(int[] nums, int left, int right) {
        if (left == right) {
            return new int[]{nums[left]};
        }
        int mid = left + (right - left) / 2;
        int[] leftArr = mergeCount(nums, left, mid);
        int[] rightArr = mergeCount(nums, mid + 1, right);
        int[] ans = new int[leftArr.length + rightArr.length];
        int cur = 0, p = 0, q = 0;
        while (p < leftArr.length && q < rightArr.length) {
            if (leftArr[p] > rightArr[q]) {
                ans[cur++] = leftArr[p];
                N += rightArr.length - q;
                p++;
            } else {
                ans[cur++] = rightArr[q++];
            }
        }
        while (p < leftArr.length) {
            ans[cur++] = leftArr[p++];
        }
        while (q < rightArr.length) {
            ans[cur++] = rightArr[q++];
        }
        return ans;
    }

    public String convert(String s, int numRows) {
        int n = s.length();
        int m = numRows;
        n = (n / (2 * m - 2) + 1) * (m - 1);
        char[][] board = new char[m][n];
        for (int i = 0; i < m; i++) {
            Arrays.fill(board[i], '*');
        }
        int[][] ways = {{1, 0}, {-1, 1}};
        int way = 0;
        int curRow = 0, curCol = 0;
        for (int i = 0; i < s.length(); i++) {
            board[curRow][curCol] = s.charAt(i);
            curRow += ways[way][0];
            curCol += ways[way][1];
            if (curRow == m) {
                way = (way + 1) % 2;
                curRow = m - 2;
                curCol++;
            }
            if (curRow < 0) {
                way = (way + 1) % 2;
                curRow = 1;
                curCol--;
            }
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] != '*') {
                    sb.append(board[i][j]);
                }
            }
        }
        return sb.toString();
    }

    int binarySearch(int[] nums, int target) {
        int left = 0, right = nums.length;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] > target) {
                right = mid;
            } else if (nums[mid] < target) {
                left = mid + 1;
            }
        }
        return -1;
    }

    public int search(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        if (nums[right] > nums[left]) {
            return binarySearch(nums, target);
        }
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (target == nums[mid]) {
                return mid;
            } else if (target < nums[mid]) {
                if (target == nums[left]) {
                    return left;
                }
                if (target == nums[right]) {
                    return right;
                } else if (target < nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            } else if (target > nums[mid]) {
                if (target == nums[right]) {
                    return right;
                }
                if (target == nums[left]) {
                    return left;
                } else if (target > nums[left]) {
                    right = mid - 1;
                } else if (target < nums[left]) {
                    left = mid + 1;
                }
            }
        }
        return -1;
    }

    public int trap(int[] height) {
        int n = height.length;
        int leftmax = height[0];
        int rightmax = height[n - 1];
        int left = 1, right = n - 2;
        int ans = 0;
        while (left <= right) {
            leftmax = Math.max(leftmax, height[left - 1]);
            rightmax = Math.max(rightmax, height[right + 1]);
            if (leftmax < rightmax) {
                ans += Math.max(leftmax - height[left], 0);
                left++;
            } else {
                ans += Math.max(rightmax - height[right], 0);
                right--;
            }
        }
        return ans;
    }

    boolean[][] dp;

    public List<List<String>> partition(String s) {
        int n = s.length();
        dp = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            dp[i][i] = true;
            if (i + 1 < n) {
                dp[i + 1][i] = true;
            }
        }
        for (int i = n - 2; i >= 0; i--) {
            for (int j = i + 1; j < n; j++) {
                dp[i][j] = dp[i + 1][j - 1] & s.charAt(i) == s.charAt(j);
            }
        }
        backtrack(s, 0);
        return trs;
    }

    LinkedList<String> tr = new LinkedList<>();
    LinkedList<List<String>> trs = new LinkedList<>();

    void backtrack(String s, int curIndex) {
        if (curIndex == dp.length) {
            trs.add(new LinkedList<>(tr));
            return;
        }
        for (int i = curIndex; i < dp.length; i++) {
            if (dp[curIndex][i]) {
                tr.add(s.substring(curIndex, i + 1));
                backtrack(s, i + 1);
                tr.removeLast();
            }
        }
    }

    public int numIslands(char[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        boolean[][] visited = new boolean[m][n];
        int ans = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (!visited[i][j] && grid[i][j] == '1') {
                    dfs(grid, visited, i, j);
                    ans++;
                }
            }
        }
        return ans;
    }

    void dfs(char[][] grid, boolean[][] visited, int x, int y) {
        int m = grid.length;
        int n = grid[0].length;
        visited[x][y] = true;
        int[] dx = {-1, 0, 1, 0};
        int[] dy = {0, 1, 0, -1};
        for (int i = 0; i < 4; i++) {
            int nx = x + dx[i];
            int ny = y + dy[i];
            if (nx >= 0 && nx < m && ny >= 0 && ny < n && !visited[nx][ny] && grid[nx][ny] == '1') {
                dfs(grid, visited, nx, ny);
            }
        }
    }

    public boolean isInterleave(String s1, String s2, String s3) {
        int m = s1.length(), n = s2.length();
        if (s3.length() != m + n) {
            return false;
        }
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int i = 1; i <= m && s1.charAt(i - 1) == s3.charAt(i - 1); i++) {
            dp[i][0] = true;
        }
        for (int i = 1; i <= n && s2.charAt(i - 1) == s3.charAt(i - 1); i++) {
            dp[0][i] = true;
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                dp[i][j] = (dp[i - 1][j] && s1.charAt(i - 1) == s3.charAt(i + j - 1)) ||
                        (dp[i][j - 1] && s2.charAt(j - 1) == s3.charAt(i + j - 1));
            }
        }
        return dp[m][n];
    }
    public int nthSuperUglyNumber(int n, int[] primes) {
        int[]p=new int[primes.length];
        int[]dp=new int[n];
        dp[0]=1;
        for(int i=1;i<n;i++){
            int min=Integer.MAX_VALUE;
            for(int j=0;j<primes.length;j++){
                if(dp[p[j]]*primes[j]<min){
                    min=dp[p[j]]*primes[j];
                }
            }
            dp[i]=min;
            for(int j=0;j<primes.length;j++){
                if(min==dp[p[j]]*primes[j]){
                    p[j]++;
                }
            }
        }
        return dp[n-1];
    }
    public int computeArea(int A, int B, int C, int D, int E, int F, int G, int H) {
        if(A>E){
            return computeArea(E,F,G,H,A,B,C,D);
        }
        int area1=(C-A)*(D-B),area2=(G-E)*(H-F);
        if(B>=H||C<=E||D<=F||area1==0||area2==0){
            return area1+area2;
        }
        int left= E;
        int right=Math.min(C,G);
        int up=Math.min(D,H);
        int down=Math.max(B,F);
        return (C-A)*(D-B)+(G-E)*(H-F)-(up-down)*(right-left);
    }
    public String getHint(String secret, String guess) {
        HashMap<Character,Integer>mp=new HashMap<>();
        HashSet<String>set=new HashSet<>();
        for(int i=0;i<secret.length();i++){
            char c=secret.charAt(i);
            set.add(c+","+i);
            mp.put(c,mp.getOrDefault(c,0)+1);
        }
        int bull=0,cow=0;
        for(int i=0;i<guess.length();i++){
            char c=guess.charAt(i);
            String key=c+","+i;
            if(set.contains(key)){
                bull++;
                mp.put(c,mp.get(c)-1);
            }
        }
        for(int i=0;i<guess.length();i++){
            char c=guess.charAt(i);
            String key=c+","+i;
            if(mp.containsKey(c)&&mp.get(c)>0&&!set.contains(key)){
                cow++;
                mp.put(c,mp.get(c)-1);
            }
        }
        return bull+"A"+cow+"B";
    }
    public int firstMissingPositive(int[] nums) {
        int N=nums.length;
        for(int i=0;i<nums.length;i++){
            if(nums[i]<=0){
                nums[i]=N+1;
            }
        }
        for(int i=0;i<nums.length;i++){
            int num=Math.abs(nums[i]);
            if(num<=N){
                nums[num-1]=-Math.abs(nums[num-1]);
            }
        }
        for(int i=0;i<nums.length;i++){
            if(nums[i]<=0){
                return i+1;
            }
        }
        return N+1;
    }
    public double myPow(double x, int n) {
        if(n==0){
            return 1;
        }
        if(n==1){
            return x;
        }
        if(n==-1){
            return 1/x;
        }
        double result=myPow(x,n/2);
        result*=result;
        if(n%2!=0){
            if(n>0) {
                result *= x;
            }else {
                result/=x;
            }
        }
        return result;
    }

    public static void main(String[] args) {
        drill dr = new drill();
        char[][] board = {{'5', '3', '.', '.', '7', '.', '.', '.', '.'},
                {'6', '.', '.', '1', '9', '5', '.', '.', '.'},
                {'.', '9', '8', '.', '.', '.', '.', '6', '.'},
                {'8', '.', '.', '.', '6', '.', '.', '.', '3'},
                {'4', '.', '.', '8', '.', '3', '.', '.', '1'},
                {'7', '.', '.', '.', '2', '.', '.', '.', '6'},
                {'.', '6', '.', '.', '.', '.', '2', '8', '.'},
                {'.', '.', '.', '4', '1', '9', '.', '.', '5'},
                {'.', '.', '.', '.', '8', '.', '.', '7', '9'}};
        dr.solveSudoku(board);
        System.out.println(Arrays.deepToString(dr.ansboard));
        int n = 12;
        int[]primes = {2,7,13,19};
        System.out.println(dr.nthSuperUglyNumber(n,primes));
        System.out.println(dr.computeArea(-3,0,3,4,0,-1,9,2));
        String secret = "1122", guess = "1222";
        System.out.println(dr.getHint(secret,guess));
    }
}
