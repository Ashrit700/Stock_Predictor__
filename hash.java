import java.util.*;
class hash{
    public static void main(String[] args){
        Scanner sc=new Scanner(System.in);
        int n=sc.nextInt();
        int[] arr=new int[n];
        for(int i=0;i<n;i++){
            arr[i]=sc.nextInt();
        }
        int pos=sc.nextInt();
        boolean t=false;
        // for(int i=0;i<n-pos;i++){
        //     for(int j=i;j<(i+pos);j++){
        //         if(arr[i]==arr[j]){
        //             System.out.println("Found");
        //             t=true;
        //         }
        //     }
        //     if(t==true){
        //         break;

        //     }

        // }
        HashMap<Integer,Integer>hash=new HashMap<>();
        for(int i=0;i<n;i++){
            if(hash.containsKey(arr[i])){
                int prev=hash.get(arr[i]);
                if(i-prev<=pos){
                     System.out.println("Found");
                   t=true;
                   break;

                }
            }
            hash.put(arr[i],i);
        }
    }
}