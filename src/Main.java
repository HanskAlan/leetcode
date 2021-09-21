
import java.util.*;
import java.lang.*;

public class Main {
    public int minSteps(int n) {
        int[]dp=new int[n+1];
        dp[1]=0;
        for(int i=2;i<=n;i++){
            dp[i]=i;
        }
        for(int i=2;i<=n;i++){
            for(int j=2*i;j<=n;j+=i){
                if(j==2*i){
                    dp[j]=Math.min(dp[j],dp[i]+2);
                }else {
                    dp[j]=Math.min(dp[j],(j-2*i)/i+dp[2*i]);
                }
            }
        }
        return dp[n];
    }
    public static void main(String[] args) {
        Main ma=new Main();
        System.out.println(ma.minSteps(3));
    }
}