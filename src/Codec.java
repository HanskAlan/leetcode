import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class Codec {

    String sep = ",";
    String NULL = "#";
    StringBuilder sb = new StringBuilder();

    public String serialize(TreeNode root) {
        StringBuilder sb=new StringBuilder();
        serialize(root,sb);
        return sb.toString();
    }

    public void serialize(TreeNode root, StringBuilder sb) {
        if (root == null) {
            sb.append(NULL).append(sep);
            return;
        }
        sb.append(root.val).append(sep);
        serialize(root.left, sb);
        serialize(root.right, sb);
    }

    public TreeNode deserialize(String data) {
        String[]datas=data.split(sep);
        LinkedList<String> list = new LinkedList<>(Arrays.asList(datas));
        return deserialize(list);
    }
    public TreeNode deserialize(LinkedList<String>list){
        if(list.isEmpty()){
            return null;
        }
        String cur=list.removeFirst();
        if(cur.equals(NULL)){
            return null;
        }
        TreeNode root=new TreeNode(Integer.parseInt(cur));
        root.left=deserialize(list);
        root.right=deserialize(list);
        return root;
    }
}