import sun.lwawt.macosx.CSystemTray;

import java.util.*;
import java.lang.*;

public class exercise {
    public boolean wordPattern(String pattern, String s) {
        String[] arr = s.split(" ");
        if (pattern.length() != arr.length) {
            return false;
        }
        int n = pattern.length();
        HashMap<Character, String> mp = new HashMap<>();
        for (int i = 0; i < n; i++) {
            if (!mp.containsKey(pattern.charAt(i))) {
                if (mp.containsValue(arr[i])) {
                    return false;
                }
                mp.put(pattern.charAt(i), arr[i]);
            } else {
                if (!mp.get(pattern.charAt(i)).equals(arr[i])) {
                    return false;
                }
            }
        }
        return true;
    }

    public int numTrees(int n) {
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i <= n; i++) {
            for (int j = 1; j <= i; j++) {
                dp[i] += dp[i - j] * dp[j - 1];
            }
        }
        return dp[n];
    }

    public ListNode insertionSortList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode q = head.next, qPre = head;
        ListNode guard = new ListNode(Integer.MIN_VALUE);
        guard.next = head;
        while (q != null) {
            if (q.val < qPre.val) {
                qPre.next = q.next;
                ListNode pt = head, ptPre = guard;
                while (pt.val < q.val) {
                    ptPre = pt;
                    pt = pt.next;
                }
                q.next = pt;
                ptPre.next = q;
                q = qPre.next;
                head = guard.next;
                continue;
            }
            qPre = q;
            q = q.next;
        }
        return guard.next;
    }

    public int minCut(String s) {
        int n = s.length();
        boolean[][] dp = new boolean[n][n];
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
        return dfs(dp, 0, n - 1);
    }

    HashMap<String, Integer> mp = new HashMap<>();

    int dfs(boolean[][] dp, int left, int right) {
        if (dp[left][right]) {
            return 0;
        }
        String key = left + "," + right;
        if (mp.containsKey(key)) {
            return mp.get(key);
        }
        int ans = Integer.MAX_VALUE;
        for (int i = left; i <= right; i++) {
            if (dp[left][i]) {
                ans = Math.min(ans, 1 + dfs(dp, i + 1, right));
            }
        }
        mp.put(key, ans);
        return ans;
    }

    public int compareVersion(String version1, String version2) {
        String[] ver1 = version1.split("\\."), ver2 = version2.split("\\.");
        int[] v1 = new int[ver1.length], v2 = new int[ver2.length];
        for (int i = 0; i < v1.length; i++) {
            v1[i] = Integer.parseInt(ver1[i]);
        }
        for (int i = 0; i < v2.length; i++) {
            v2[i] = Integer.parseInt(ver2[i]);
        }
        int len = Math.min(v1.length, v2.length);
        for (int i = 0; i < len; i++) {
            if (v1[i] > v2[i]) {
                return 1;
            } else if (v1[i] < v2[i]) {
                return -1;
            }
        }
        for (int i = len; i < v1.length; i++) {
            if (v1[i] != 0) {
                return 1;
            }
        }
        for (int i = len; i < v2.length; i++) {
            if (v2[i] != 0) {
                return -1;
            }
        }
        return 0;
    }

    int getMin(int a, int b, int c) {
        return Math.min(a, Math.min(b, c));
    }

    int generateMask(String word) {
        int mask = 0;
        for (int i = 0; i < word.length(); i++) {
            int n = word.charAt(i) - 'a';
            mask |= (1 << n);
        }
        return mask;
    }

    public int maxProduct(String[] words) {
        HashMap<Integer, Integer> mp = new HashMap<>();
        for (String word : words) {
            int mask = generateMask(word);
            mp.put(mask, Math.max(word.length(), mp.getOrDefault(mask, 0)));
        }
        int prod = 0;
        for (int i : mp.keySet()) {
            for (int j : mp.keySet()) {
                if ((i & j) == 0) {
                    prod = Math.max(prod, mp.get(i) * mp.get(j));
                }
            }
        }
        return prod;
    }

    public boolean isAdditiveNumber(String num) {
        int n = num.length();
        for (int i = 1; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (num.charAt(i) == '0' && j - i > 1) {
                    continue;
                }
                if (dfs(num, 0, i, j)) {
                    return true;
                }
            }
        }
        return false;
    }

    boolean dfs(String num, int seg1, int seg2, int seg3) {
        if (seg3 == num.length()) {
            return true;
        }
        for (int i = seg3 + 1; i <= num.length(); i++) {
            String s1 = num.substring(seg1, seg2);
            String s2 = num.substring(seg2, seg3);
            String s3 = num.substring(seg3, i);
            if ((s1.charAt(0) == '0' && s1.length() > 1) || (s2.charAt(0) == '0' && s2.length() > 1) ||
                    (s3.charAt(0) == '0' && s3.length() > 1)) {
                continue;
            }
            if (Integer.parseInt(s1) + Integer.parseInt(s2) == Integer.parseInt(s3)) {
                if (dfs(num, seg2, seg3, i)) {
                    return true;
                }
            }
        }
        return false;
    }

    public void wiggleSort(int[] nums) {
        int[] help = nums.clone();
        Arrays.sort(help);
        int n = nums.length;
        for (int i = 1; i < n; i += 2) {
            nums[i] = help[--n];
        }
        for (int i = 0; i < n; i += 2) {
            nums[i] = help[--n];
        }
    }

    public int bulbSwitch(int n) {
        int on = 0;
        for (int i = 1; i <= n; i++) {
            int sw = 0;
            for (int j = 1; j <= i; j++) {
                if (i % j == 0) {
                    sw++;
                }
            }
            if (sw % 2 == 1) {
                on++;
            }
        }
        return on;
    }

    public int lengthOfLongestSubstring(String s) {
        int n = s.length();
        int left = 0, right = 0;
        int maxLen = -1;
        LinkedList<Character> window = new LinkedList<>();
        HashMap<Character, Integer> mp = new HashMap<>();
        while (right < n) {
            char c = s.charAt(right);
            window.add(c);
            mp.put(c, mp.getOrDefault(c, 0) + 1);
            right++;
            while (mp.get(c) > 1) {
                maxLen = Math.max(maxLen, window.size() - 1);
                char temp = window.getFirst();
                window.removeFirst();
                mp.put(temp, mp.get(temp) - 1);
                if (mp.get(temp) == 0) {
                    mp.remove(temp);
                }
            }
            if (right == n) {
                maxLen = Math.max(maxLen, window.size());
            }
        }
        return maxLen;
    }

    public int candy(int[] ratings) {
        int n = ratings.length;
        int[] arr = new int[n];
        Arrays.fill(arr, 1);
        int sum = n;
        for (int i = 0; i < n - 1; i++) {
            while (ratings[i] < ratings[i + 1] && arr[i] >= arr[i + 1]) {
                arr[i + 1]++;
                sum++;
            }
        }
        for (int i = n - 1; i >= 1; i--) {
            while (ratings[i] < ratings[i - 1] && arr[i] >= arr[i - 1]) {
                arr[i - 1]++;
                sum++;
            }
        }
        return sum;
    }

    public int maxPoints(int[][] points) {
        Arrays.sort(points, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] - o2[0];
            }
        });
        for(int i=0;i<points.length;i++){
            LinkedList<Integer>line=new LinkedList<>();
            line.add(i);
            backtrack(points,i,line);
        }
        return maxLen;
    }
    int maxLen=-1;
    void backtrack(int[][]points,int p1,LinkedList<Integer>line){
        maxLen=Math.max(maxLen,line.size());
        for(int i=p1+1;i<points.length;i++){
            if(line.size()<2){
                line.add(i);
                backtrack(points,i,line);
                line.removeLast();
            }else{
                int p=line.getFirst();
                int q=line.getLast();
                if((points[q][1]-points[p][1])*(points[i][0]-points[q][0])==
                        (points[q][0]-points[p][0])*(points[i][1]-points[q][1])){
                    line.add(i);
                    backtrack(points,i,line);
                    line.removeLast();
                }
            }
        }
    }
    public int longestValidParentheses(String s) {
        int n=s.length();
        Stack<Character>st=new Stack<>();
        int maxLen=-1;
        for(int i=0;i<n;i++){
            if(s.charAt(i)=='('){
                st.clear();
                st.push('(');
                for(int j=i+1;j<n;j++){
                    if(s.charAt(j)=='('){
                        st.push('(');
                    }else{
                        if(st.isEmpty()){
                            break;
                        }
                        st.pop();
                        if(st.isEmpty()){
                            maxLen=Math.max(maxLen,j-i+1);
                        }
                    }
                }
            }
        }
        return maxLen;
    }
    public List<String> fullJustify(String[] words, int maxWidth) {
        int n=words.length;
        int curLen=0;
        LinkedList<String>line=new LinkedList<>();
        List<String>ans=new LinkedList<>();
        for(int i=0;i<n;i++){
            if(curLen+words[i].length()+line.size()<=maxWidth){
                line.add(words[i]);
                curLen+=words[i].length();
            }else{
                ans.add(fulling(line,maxWidth));
                line.clear();
                line.add(words[i]);
                curLen=words[i].length();
            }
        }
        if(line.size()!=0){
            ans.add(leftFulling(line,maxWidth));
        }
        return ans;
    }
    public String fulling(LinkedList<String>line,int maxLen){
        int len=line.size()-1;//单词间隔数
        int totalLen=0;
        for(String s:line){
            totalLen+=s.length();
        }
        int tabs=maxLen-totalLen;
        int basicLen;
        if(len!=0) {
            basicLen = tabs / len;
        }else{
            basicLen=0;
        }
        int left;
        if(len==0){
            left=0;
        }else {
            left = tabs % len;
        }
        String ans="";
        for(int i=0;i<left;i++){
            ans+=line.get(i);
            for(int j=0;j<basicLen+1;j++){
                ans+=" ";
            }
        }
        for(int i=left;i<line.size();i++){
            ans+=line.get(i);
            if(i!=line.size()-1) {
                for (int j = 0; j < basicLen; j++) {
                    ans += " ";
                }
            }
        }
        if(len==0){
            for(int i=0;i<maxLen-line.getFirst().length();i++){
                ans+=" ";
            }
        }
        return ans;
    }
    public String leftFulling(LinkedList<String>line ,int maxLen){
        String ans="";
        for(int i=0;i<line.size();i++){
            ans+=line.get(i);
            if(i!=line.size()-1){
                ans+=" ";
            }
        }
        int left=maxLen-ans.length();
        for(int i=0;i<left;i++){
            ans+=" ";
        }
        return ans;
    }


    public int maxCoins(int[] nums) {
        int n=nums.length;
        int[]points=new int[n+2];
        points[0]=points[n+1]=1;
        for(int i=1;i<=n;i++){
            points[i]=nums[i-1];
        }
        int[][]dp=new int[n+2][n+2];
        for(int i=n;i>=0;i--){
            for(int j=i+1;j<n+2;j++){
                for(int k=i+1;k<j;k++){
                    dp[i][j]=Math.max(dp[i][j],dp[i][k]+dp[k][j]+points[i]*points[k]*points[j]);
                }
            }
        }
        return dp[0][n+1];
    }
    public int numberOfArithmeticSlices(int[] A) {
        int n=A.length;
        for(int i=0;i<=n-3;i++){
            LinkedList<Integer>chosen=new LinkedList<>();
            chosen.add(A[i]);
            dfs(A,chosen,i);
        }
        System.out.println(ans);
        return ans.size();
    }
    List<List<Integer>>ans=new LinkedList<>();
    void dfs(int[]A,LinkedList<Integer>chosen,int cur){
        if(cur==A.length){
            return;
        }
        for(int i=cur+1;i<A.length;i++){
            if(chosen.size()==1){
                chosen.add(A[i]);
                dfs(A,chosen,i);
                chosen.removeLast();
            }else{
                int left=chosen.getLast()-chosen.get(chosen.size()-2);
                if(A[i]-chosen.getLast()>left){
                    if(chosen.size()>2){
                        break;
                    }
                }else if(A[i]-chosen.getLast()==left){
                    chosen.add(A[i]);
                    ans.add(new LinkedList<>(chosen));
                    dfs(A,chosen,i);
                    chosen.removeLast();
                }
            }
        }
    }
    public int islandPerimeter(int[][] grid) {
        int[][]dir={{-1,0,1,0},{0,1,0,-1}};
        int ans=0;
        int m=grid.length,n=grid[0].length;
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(grid[i][j]==1) {
                    for (int k = 0; k < 4; k++) {
                        int ni = i + dir[0][k];
                        int nj = j + dir[1][k];
                        if (ni < 0 || ni >= m || nj < 0 || nj >= n || grid[ni][nj] == 0) {
                            ans++;
                        }
                    }
                }
            }
        }
        return ans;
    }
    public int maxPathSum(TreeNode root) {
        maxGain(root);
        return maxSum;
    }
    int maxSum=Integer.MIN_VALUE;
    int maxGain(TreeNode root){
        if(root==null){
            return 0;
        }
        int leftGain=Math.max(maxGain(root.left),0);
        int rightGain=Math.max(maxGain(root.right),0);
        int pathValue=root.val+leftGain+rightGain;
        maxSum=Math.max(maxSum,pathValue);
        return root.val+Math.max(leftGain,rightGain);
    }
    public int nthSuperUglyNumber(int n, int[] primes) {
        if(n==1){
            return 1;
        }
        HashSet<Integer>prime=new HashSet<>();
        HashSet<Integer>ugly=new HashSet<>();
        for(int p:primes){
            prime.add(p);
            ugly.add(p);
        }
        ugly.add(1);
        int step=1;
        int ans=-1;
        for(int i=2;i<Integer.MAX_VALUE;i++){
            //System.out.println("sdfdsfds "+step);
            if(ugly.contains(i)){
                step++;
                if(step==n){
                    ans=i;
                    break;
                }
            }else{
                for(int p:prime){
                    if(i%p==0&&ugly.contains(i/p)){
                        ugly.add(i);
                        step++;
                        if(step==n){
                            ans=i;
                            return ans;
                        }
                        break;
                    }
                }
            }
        }
        return ans;
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
    List<String>getNeighbor(String word,HashSet<String>List,HashSet<String>visited){
        List<String>ans=new LinkedList<>();
        char[]arr=word.toCharArray();
        int len=arr.length;
        for(int i=0;i<len;i++){
            char temp=arr[i];
            for(char c='a';c<='z';c++){
                if(temp==c){
                    continue;
                }
                arr[i]=c;
                String cur=new String(arr);
                if(List.contains(cur)&&!visited.contains(cur)){
                    ans.add(cur);
                }
            }
            arr[i]=temp;
        }
        return ans;
    }


    String getMoves(char[][]dict,String input){
        int n=input.length();
        int i=0;
        char start='a',target=input.charAt(i);
        String ans="";
        while(i<n){
            int lenStart=start-'a';
            int sx=lenStart/5;
            int sy=lenStart%5;
            int lenTarget=target-'a';
            int tx=lenTarget/5;
            int ty=lenTarget%5;
            int dx=tx-sx;
            int dy=ty-sy;
            if(dx<0){
                for(int j=0;j<Math.abs(dx);j++){
                    ans+="L";
                }
            }else{
                for(int j=0;j<Math.abs(dx);j++){
                    ans+="R";
                }
            }
            if(dy<0){
                for(int j=0;j<Math.abs(dy);j++){
                    ans+="D";
                }
            }else{
                for(int j=0;j<Math.abs(dy);j++){
                    ans+="U";
                }
            }
            ans+="!";
            start=target;
            i++;
            target=input.charAt(i);
        }
        return ans;
    }

    public static void main(String[] args) {
        exercise ex = new exercise();
        String pattern = "abba", str = "dog dog dog dog";
        System.out.println(ex.wordPattern(pattern, str));
        System.out.println(ex.numTrees(3));
        int[] arr = {-1, 5, 3, 4, 0};
        ListNode head = new ListNode(arr);
        System.out.println(ex.insertionSortList(head));
        String s = "abab";
        System.out.println(ex.minCut(s));
        String version1 = "7.5.2.4", version2 = "7.5.3";
        System.out.println(ex.compareVersion(version1, version2));
        String[] words = {"abcw", "baz", "foo", "bar", "xtfn", "abcdef"};
        System.out.println(ex.maxProduct(words));
        String beginWord = "hit", endWord = "cog";
        String[]wordList = {"hot","dot","dog","lot","log","cog"};
        List<String> wordlist = new LinkedList<>(Arrays.asList(wordList));


    }
}
