import java.util.Comparator;
import java.util.PriorityQueue;

public class MedianFinder {
    PriorityQueue<Integer> smaller, larger;//小数应该是大顶堆

    public MedianFinder() {
        smaller = new PriorityQueue<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return o2 - o1;
            }
        });
        larger = new PriorityQueue<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return o1 - o2;
            }
        });
    }

    public void addNum(int num) {
        if (smaller.isEmpty() || num > smaller.peek()) {
            larger.add(num);
            if(larger.size()>smaller.size()){
                smaller.add(larger.poll());
            }
        }else {
            smaller.add(num);
            if(smaller.size()- larger.size()>1){
                larger.add(smaller.poll());
            }
        }
    }

    public double findMedian() {
        if (smaller.isEmpty()) {
            return 0;
        }
        if (smaller.size() > larger.size()) {
            return smaller.peek();
        } else {
            return (double) (smaller.peek() + larger.peek()) / 2;
        }
    }

    public static void main(String[] args) {
        MedianFinder obj = new MedianFinder();
        obj.addNum(3);
        double a = obj.findMedian();
        obj.addNum(2);
        double b = obj.findMedian();
        System.out.println(a + " " + b);
    }
}
