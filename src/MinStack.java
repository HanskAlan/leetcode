import java.util.*;
import java.lang.*;
class MinStack {
    Stack<int[]>st;
    /** initialize your data structure here. */
    public MinStack() {
        st=new Stack<>();
    }

    public void push(int x) {
        if(st.isEmpty()){
            st.push(new int[]{x,x});
        }else {
            int min=Math.min(st.peek()[1],x);
            st.push(new int[]{x,min});
        }
    }

    public void pop() {
        st.pop();
    }

    public int top() {
        return st.peek()[0];
    }

    public int getMin() {
        return st.peek()[1];
    }
    public static void main(String[]args){
        MinStack ms=new MinStack();
        ms.push(3);
        ms.push(2);
        ms.pop();
        int param_3 = ms.top();
        int param_4 = ms.getMin();
        System.out.println(param_3+" "+param_4);
    }
}
