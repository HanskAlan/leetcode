import java.util.*;

class LRUCache {
    class DLinkedNode{
        int key;
        int val;
        DLinkedNode prev;
        DLinkedNode next;
        DLinkedNode(int key,int val){
            this.key=key;
            this.val=val;
        }
    }
    int capacity;
    DLinkedNode head;
    DLinkedNode tail;
    HashMap<Integer,DLinkedNode>cache;
    int size;

    public LRUCache(int capacity) {
        this.capacity=capacity;
        this.size=0;
        this.cache=new HashMap<>();
        this.head=new DLinkedNode(-1,-1);
        this.tail=new DLinkedNode(-1,-1);
        head.next=tail;
        tail.prev=head;
    }

    public int get(int key) {
        if(!cache.containsKey(key)){
            return -1;
        }
        removeNode(key);
        addNodeAtHead(key,cache.get(key).val);
        return cache.get(key).val;
    }
    void removeNode(int key){
        DLinkedNode node=cache.get(key);
        node.prev.next=node.next;
        node.next.prev=node.prev;
    }
    void removeNodeAtTail(){
        int key=tail.prev.key;
        cache.remove(key);
        tail.prev=tail.prev.prev;
        tail.prev.next=tail;
    }
    void addNodeAtHead(int key,int value){
        DLinkedNode node=new DLinkedNode(key,value);
        node.next=head.next;
        node.next.prev=node;
        head.next=node;
        node.prev=head;
        cache.put(key,node);
    }
    public void put(int key, int value) {
        if(cache.containsKey(key)){
            cache.get(key).val=value;
            removeNode(key);
            addNodeAtHead(key,value);
        }else {
            addNodeAtHead(key,value);
            size++;
            while(size>capacity){
                removeNodeAtTail();
                size--;
            }
        }
    }
    public int countDigitOne(int n) {
        if(n==0){
            return 0;
        }
        int digit=1;
        int cur=n%10;
        int high=n/10;
        int low=0;
        int ans=0;
        while (high!=0||cur!=0){
            if(cur==0){
                ans+=high*digit;
            }else if(cur==1){
                ans+=high*digit+low+1;
            }else {
                ans+=(high+1)*digit;
            }
            low+=cur*digit;
            cur=high%10;
            digit*=10;
            high/=10;
        }
        return ans;
    }
    public int balancedStringSplit(String s) {
        int n=s.length();
        int lnum=0,rnum=0;
        int ans=0;
        for(int i=0;i<n;i++){
            if(s.charAt(i)=='L'){
                lnum++;
            }else {
                rnum++;
            }
            if(lnum==rnum){
                ans++;
                lnum=0;
                rnum=0;
            }
        }
        return ans;
    }
    public int wiggleMaxLength(int[] nums) {
        int n=nums.length;
        int[][]dp=new int[n][2];
        dp[0][0]=1;
        dp[0][1]=0;
        int max=-1;
        for(int i=1;i<n;i++){
            for(int j=0;j<i;j++){
                if(nums[j]>nums[i]&&(dp[j][1]==0||dp[j][1]==1)){
                    if(dp[j][0]+1>dp[i][0]){
                        dp[i][0]=dp[j][0]+1;
                        dp[i][1]=-1;
                        max=Math.max(max,dp[i][0]);
                    }
                }else if(nums[j]<nums[i]&&(dp[j][1]==0||dp[j][1]==-1)){
                    if(dp[j][0]+1>dp[i][0]){
                        dp[i][0]=dp[j][0]+1;
                        dp[i][1]=1;
                        max=Math.max(max,dp[i][0]);
                    }
                }
            }
        }
        return max;
    }
    public List<Integer> largestDivisibleSubset(int[] nums) {
        LinkedList<Integer>ans=new LinkedList<>();
        Arrays.sort(nums);
        int n=nums.length;
        int max=1;
        int last=0;
        int[][]dp=new int[n][2];
        for(int i=0;i<n;i++){
            dp[i][1]=-1;
        }
        for(int i=0;i<n;i++){
            for(int j=0;j<i;j++){
                if(nums[i]%nums[j]==0){
                    if(dp[j][0]==0){
                        dp[j][0]=1;
                        dp[i][0]=2;
                        dp[i][1]=j;
                        continue;
                    }
                    if(dp[j][0]+1>dp[i][0]){
                        dp[i][0]=dp[j][0]+1;
                        dp[i][1]=j;
                        if(dp[i][0]>max){
                            max=dp[i][0];
                            last=i;
                        }
                    }
                }
            }
        }
        int p=last;
        while (dp[p][1]!=-1){
            ans.addFirst(nums[p]);
            p=dp[p][1];
        }
        return ans;
    }
    public static void main(String[] args) {
        LRUCache lRUCache = new LRUCache(2);
        lRUCache.put(2, 1);
        lRUCache.put(1, 1);
        lRUCache.put(2, 3);
        lRUCache.put(4, 1);
        System.out.println(lRUCache.get(1));    // 返回 -1 (未找到)
        System.out.println(lRUCache.get(2));    // 返回 3
        System.out.println(lRUCache.countDigitOne(13));
        int[]nums = {1,2,3,8};
        System.out.println(lRUCache.wiggleMaxLength(nums));
        System.out.println(lRUCache.largestDivisibleSubset(nums));
    }
}