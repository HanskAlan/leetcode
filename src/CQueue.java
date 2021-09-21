import java.util.Stack;

class CQueue {
    Stack<Integer>st1,st2;
    public CQueue() {
        st1=new Stack<>();
        st2=new Stack<>();
    }

    public void appendTail(int value) {
        st1.push(value);
    }

    public int deleteHead() {
        if(!st2.isEmpty()){
            return st2.pop();
        }
        if(st1.isEmpty()){
            return -1;
        }
        while (!st1.isEmpty()){
            st2.push(st1.pop());
        }
        return st2.pop();
    }
}
