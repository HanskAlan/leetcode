import java.util.ArrayDeque;
import java.util.Deque;
import java.util.LinkedList;
import java.util.Queue;

public class MaxQueue {
    Queue<Integer> q;
    Deque<Integer> dq;

    public MaxQueue() {
        q = new LinkedList<>();
        dq = new ArrayDeque<>();
    }

    public int max_value() {
        if (!dq.isEmpty()) {
            return dq.getFirst();
        }
        return -1;
    }

    public void push_back(int value) {
        while (!dq.isEmpty() && dq.getLast() < value) {
            dq.pollLast();
        }
        dq.offerLast(value);
        q.offer(value);
    }

    public int pop_front() {
        if (q.isEmpty()) {
            return -1;
        }
        int ans=q.poll();
        if(ans==dq.peekFirst()){
            dq.pollFirst();
        }
        return ans;
    }

    public static void main(String[] args) {
        MaxQueue mq = new MaxQueue();
        mq.push_back(1);
        mq.push_back(2);
        System.out.println(mq.max_value());
        System.out.println(mq.pop_front());
        System.out.println(mq.max_value());
    }
}
