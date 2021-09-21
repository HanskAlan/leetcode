import java.util.*;

public class MyLinkedList {
    class ListNode {
        int val;
        ListNode next;
        ListNode prev;

        ListNode(int x) {
            val = x;
        }
    }

    ListNode head;
    ListNode tail;
    int size;

    /**
     * Initialize your data structure here.
     */
    public MyLinkedList() {
        head = new ListNode(-1);
        tail = new ListNode(-1);
        head.next = tail;
        tail.prev = head;
        size = 0;
    }

    /**
     * Get the value of the index-th node in the linked list. If the index is invalid, return -1.
     */
    public int get(int index) {
        if (index >= size || index < 0) {
            return -1;
        }
        ListNode pivot = head.next;
        for (int i = 0; i < index; i++) {
            pivot = pivot.next;
        }
        return pivot.val;
    }

    /**
     * Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list.
     */
    public void addAtHead(int val) {
        ListNode cur = new ListNode(val);
        cur.next = head.next;
        head.next.prev = cur;
        cur.prev = head;
        head.next = cur;
        size++;
    }

    /**
     * Append a node of value val to the last element of the linked list.
     */
    public void addAtTail(int val) {
        ListNode cur = new ListNode(val);
        cur.prev = tail.prev;
        tail.prev.next = cur;
        cur.next = tail;
        tail.prev = cur;
        size++;
    }

    /**
     * Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted.
     */
    public void addAtIndex(int index, int val) {
        if (index <= 0) {
            addAtHead(val);
        } else if (index == size) {
            addAtTail(val);
        } else if (index > size) {
            return;
        } else {
            ListNode cur = head;
            for (int i = 0; i < index; i++) {
                cur = cur.next;
            }
            ListNode node = new ListNode(val);
            node.next = cur.next;
            cur.next.prev = node;
            cur.next = node;
            node.prev = cur;
            size++;
        }
    }

    /**
     * Delete the index-th node in the linked list, if the index is valid.
     */
    public void deleteAtIndex(int index) {
        if (index < 0 || index > size) {
            return;
        }
        ListNode cur = head;
        for (int i = 0; i < index; i++) {
            cur = cur.next;
        }
        cur.next = cur.next.next;
        cur.next.prev = cur;
        size--;
    }

    public int distributeCandies(int[] candyType) {
        int n = candyType.length;
        HashSet<Integer> s = new HashSet<>();
        for (int i : candyType) {
            s.add(i);
        }
        int m = s.size();
        if (m >= n / 2) {
            return n / 2;
        } else {
            return m;
        }
    }

    public int divide(int dividend, int divisor) {
        long a = dividend;
        long b = divisor;
        int flag1 = 1, flag2 = 1;
        if (a < 0) {
            a = -a;
            flag1 = -1;
        }
        if (b < 0) {
            b = -b;
            flag2 = -1;
        }
        int flag = flag1 * flag2;
        return (int) (div(a, b) * flag);
    }

    public long div(long a, long b) {
        if (a < b) {
            return 0;
        }
        long tb = b;
        long count = 1;
        while (tb + tb <= a) {
            count += count;
            tb += tb;
        }
        return count + div(a - tb, b);
    }

    public boolean isValidSudoku(char[][] board) {
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (Character.isDigit(board[i][j]) && !isValid(board, i, j, board[i][j])) {
                    return false;
                }
            }
        }
        return true;
    }

    public boolean isValid(char[][] board, int x, int y, char c) {
        for (int i = 0; i < 9; i++) {
            if (board[x][i] == c && i != y) {
                return false;
            }
        }
        for (int i = 0; i < 9; i++) {
            if (board[i][y] == c && i != x) {
                return false;
            }
        }
        for (int i = (x / 3) * 3; i < (x / 3) * 3 + 3; i++) {
            for (int j = (y / 3) * 3; j < (y / 3) * 3 + 3; j++) {
                if (board[i][j] == c && i != x && j != y) {
                    return false;
                }
            }
        }
        return true;
    }

    public List<List<Integer>> fourSum(int[] nums, int target) {
        Arrays.sort(nums);
        return nSum(nums, target, 0,4);
    }

    public List<List<Integer>> nSum(int[] nums, int target, int start,int n) {//调用该函数前要对数组进行排序
        List<List<Integer>> ans = new LinkedList<>();
        int sz = nums.length;
        if (n < 2 || sz < n) {
            return ans;
        }
        if (n == 2) {
            int left = start, right = sz - 1;
            while (left < right) {
                int l = nums[left], r = nums[right];
                if (l + r < target) {
                    while (left<right&&nums[left]==l)left++;
                } else if (l + r > target) {
                    while (left<right&&nums[right]==r)right--;
                } else {
                    List<Integer> tmp = new LinkedList<>();
                    tmp.add(l);
                    tmp.add(r);
                    ans.add(new LinkedList<>(tmp));
                    while (nums[left] == l && left < right) {
                        left++;
                    }
                    while (nums[right] == r && left < right) {
                        right--;
                    }
                }
            }
        } else {
            for (int i = start; i < sz; i++) {
                List<List<Integer>> res = nSum(nums, target - nums[i], i+1,n - 1);
                for (List<Integer> r : res) {
                    r.add(nums[i]);
                    ans.add(new LinkedList<>(r));
                }
                while (i<sz-1&&nums[i]==nums[i+1]){
                    i++;
                }
            }
        }
        return ans;
    }
    public boolean search(int[] nums, int target) {
        int n=nums.length;
        int left=0,right=n-1;
        while (left<=right){
            int mid=left+(right-left)/2;
            if(nums[mid]==target){
                return true;
            }else if(nums[mid]==nums[left]&&nums[left]==nums[right]){
                left++;
                right--;
            }else if(nums[mid]>=nums[left]){
                if(target>=nums[left]&&target<nums[mid]){
                    right=mid-1;
                }else {
                    left=mid+1;
                }
            }else{
                if(target>nums[mid]&&target<=nums[n-1]){
                    left=mid+1;
                }else {
                    right=mid-1;
                }
            }
        }
        return false;
    }
    public void recoverTree(TreeNode root) {
        List<Integer>result=new LinkedList<>();
        inorder(root,result);
        int n=result.size();
        int x=-1,y=-1;
        boolean changed=false;
        for(int i=0;i<n-1;i++){
            if(result.get(i)>result.get(i+1)){
                if(!changed){
                    x= result.get(i);
                    y=result.get(i+1);
                    changed=true;
                }else {
                    y=result.get(i+1);
                    break;
                }
            }
        }
        recoverTree(root,x,y);
    }
    void recoverTree(TreeNode root,int x,int y){
        if(root==null){
            return;
        }
        recoverTree(root.left,x,y);
        if(root.val==x){
            root.val=y;
        }else if(root.val==y){
            root.val=x;
        }
        recoverTree(root.right,x,y);
    }
    void inorder(TreeNode root,List<Integer>result){
        if(root==null){
            return;
        }
        inorder(root.left,result);
        result.add(root.val);
        inorder(root.right,result);
    }
    public int[][] insert(int[][] intervals, int[] newInterval) {
        List<int[]> ans = new LinkedList<>();
        boolean placed=false;
        int left=newInterval[0];
        int right=newInterval[1];
        for(int[]interval:intervals){
            if(interval[0]>right){
                if(!placed){
                    ans.add(new int[]{left,right});
                    placed=true;
                }
                ans.add(interval);
            }else if(interval[1]<left){
                ans.add(interval);
            }else {
                left=Math.min(left,interval[0]);
                right=Math.max(right,interval[1]);
            }
        }
        if(!placed){
            ans.add(new int[]{left,right});
        }
        int[][]res=new int[ans.size()][];
        for(int i=0;i<ans.size();i++){
            res[i]=ans.get(i);
        }
        return res;
    }
    public List<Integer> getRow(int rowIndex) {
        List<Integer>ans=new LinkedList<>();
        ans.add(1);
        if(rowIndex==0){
            return ans;
        }
        ans.add(1);
        if(rowIndex==1){
            return ans;
        }
        int[]res={1,1};
        for(int i=0;i<rowIndex-1;i++){
            int[]tmp=res.clone();
            res=new int[tmp.length+1];
            res[0]=1;
            res[res.length-1]=1;
            for(int j=1;j<res.length-1;j++){
                res[j]=tmp[j-1]+tmp[j];
            }
        }
        ans=new LinkedList<>();
        for(int i=0;i<res.length;i++){
            ans.add(res[i]);
        }
        return ans;
    }
    public String decodeString(String s) {
        int n=s.length();
        Stack<Integer>num_st=new Stack<>();
        Stack<String>str_st=new Stack<>();
        int num=0;
        String res="";
        for(int i=0;i<n;i++){
            char c=s.charAt(i);
            if(Character.isDigit(c)){
                num=num*10+c-'0';
            }else if(Character.isAlphabetic(c)){
                res=res+c;
            }else if(c=='['){
                num_st.push(num);
                str_st.push(res);
                num=0;
                res="";
            }else if(c==']'){
                int times=num_st.pop();
                StringBuilder tmp= new StringBuilder(str_st.pop());
                for(int j=0;j<times;j++){
                    tmp.append(res);
                }
                res= tmp.toString();
            }
        }
        return res;
    }
    public int[] decode(int[] encoded, int first) {
        int n=encoded.length;
        int[]ans=new int[n+1];
        ans[0]=first;
        for(int i=1;i<=n;i++){
            ans[i]=encoded[i-1]^ans[i-1];
        }
        return ans;
    }
    public int lengthOfLongestSubstringTwoDistinct(String s) {
        int n=s.length();
        int left=0,right=0;
        HashMap<Character,Integer>map=new HashMap<>();
        HashSet<Character>set=new HashSet<>();
        LinkedList<Character>window=new LinkedList<>();
        int ans=0;
        while (right<n){
            char r=s.charAt(right++);
            window.add(r);
            map.put(r,map.getOrDefault(r,0)+1);
            while(map.size()>2){
                char l=s.charAt(left++);
                window.removeFirst();
                ans=Math.max(ans,window.size());
                map.put(l,map.get(l)-1);
                if(map.get(l)==0){
                    map.remove(l);
                }
            }
        }
        ans=Math.max(ans,right-left);
        return ans;
    }
    public List<Integer> diffWaysToCompute(String expression) {
        if(isNum(expression)){
            List<Integer>tmp=new LinkedList<>();
            tmp.add(toNum(expression));
            return tmp;
        }
        List<Integer>ans=new LinkedList<>();
        for(int i=0;i<expression.length();i++){
            char c=expression.charAt(i);
            if(c=='+'||c=='-'||c=='*'){
                List<Integer>left=diffWaysToCompute(expression.substring(0,i));
                List<Integer>right=diffWaysToCompute(expression.substring(i+1));
                for(int l:left){
                    for(int r:right){
                        if(c=='+'){
                            ans.add(l+r);
                        }else if(c=='-'){
                            ans.add(l-r);
                        }else {
                            ans.add(l*r);
                        }
                    }
                }
            }
        }
        return ans;
    }
    public int toNum(String num){
        int ans=0;
        for(int i=0;i<num.length();i++){
            ans=ans*10+num.charAt(i)-'0';
        }
        return ans;
    }
    public boolean isNum(String s){
        for(char c:s.toCharArray()){
            if(!Character.isDigit(c)){
                return false;
            }
        }
        return true;
    }


    public static void main(String[] args) {
        MyLinkedList linkedList = new MyLinkedList();
        linkedList.addAtHead(1);
        linkedList.addAtTail(3);
        linkedList.addAtIndex(1, 2);   //链表变为1-> 2-> 3
        System.out.println(linkedList.get(1));            //返回2
        linkedList.deleteAtIndex(1);  //现在链表是1-> 3



    }
}
