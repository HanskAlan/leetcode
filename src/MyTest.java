import java.util.*;
public class MyTest {
    public ListNode deleteNth(ListNode head,int n){
        if(head==null){
            return head;
        }
        ListNode p=head;
        for(int i=0;i<n;i++){
            if(p!=null) {
                p = p.next;
            }
        }
        ListNode q=head,pre=null;
        while(p!=null){
            pre=q;
            q=q.next;
            p=p.next;
        }
        pre.next=q.next;
        return head;
    }
    public List<List<Integer>>sTraverse(TreeNode root){
        Deque<TreeNode>dq=new LinkedList<>();
        List<List<Integer>>ans=new LinkedList<>();
        dq.offer(root);
        int step=0;
        while (!dq.isEmpty()){
            int sz=dq.size();
            List<Integer>curLevel=new LinkedList<>();
            for(int i=0;i<sz;i++) {
                TreeNode cur;
                if (step % 2 == 0) {
                    cur = dq.pollFirst();
                } else {
                    cur = dq.pollLast();
                }
                curLevel.add(cur.val);
                if(cur.left!=null){
                    dq.offer(cur.left);
                }
                if(cur.right!=null){
                    dq.offer(cur.right);
                }
            }
            ans.add(curLevel);
            step++;
        }
        return ans;
    }
    public static void main(String[] args) {
        Scanner in=new Scanner(System.in);
        int T=in.nextInt();
        for(int i=0;i<T;i++){
            int n=in.nextInt();
            int k=in.nextInt();
            LinkedList<Integer>arr=new LinkedList<>();
            for(int j=0;j<n;j++){
                arr.add(in.nextInt());
            }
            int cur=0;
            for(int j=arr.get(0)+1;;j++){
                if(!arr.contains(j)){
                    cur++;
                    if(cur==k){
                        System.out.println(j);
                        break;
                    }
                }
            }
        }
    }
}
