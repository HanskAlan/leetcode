public class Trie {

    /** Initialize your data structure here. */
    class TrieNode{
        public boolean isEnd;
        public TrieNode[]next;
        public TrieNode(){
            isEnd=false;
            next=new TrieNode[26];
        }
    }
    public TrieNode root;
    public Trie() {
        root=new TrieNode();
    }

    /** Inserts a word into the trie. */
    public void insert(String word) {
        TrieNode node=root;
        for(char c:word.toCharArray()){
            if(node.next[c-'a']==null){
                node.next[c-'a']=new TrieNode();
            }
            node=node.next[c-'a'];
        }
        node.isEnd=true;
    }

    /** Returns if the word is in the trie. */
    public boolean search(String word) {
        TrieNode node=root;
        for(char c:word.toCharArray()){
            if(node.next[c-'a']==null){
                return false;
            }
            node=node.next[c-'a'];
        }
        return node.isEnd;
    }

    /** Returns if there is any word in the trie that starts with the given prefix. */
    public boolean startsWith(String prefix) {
        TrieNode node=root;
        for(char c:prefix.toCharArray()){
            if(node.next[c-'a']==null){
                return false;
            }
            node=node.next[c-'a'];
        }
        return true;
    }
//    public static void main(String[]args){
//        Trie t=new Trie();
//        t.insert("hahaha");
//        System.out.println(t.search("haha"));
//        System.out.println(t.startsWith("hahahah"));
//    }
}