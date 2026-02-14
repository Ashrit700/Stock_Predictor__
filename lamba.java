import java.util.*;


class lamba{
    interface  sum{
    void add(int x,int y);

}
    public static void main(String []args){
        Scanner sc=new Scanner(System.in);
        int x=sc.nextInt();
        int y=sc.nextInt();
        sum s=(a,b)->System.out.println(a+b);
        s.add(x,y);
        Function<Integer,Integer>square=c->c*c;
        System.out.println(square.apply(3));


            }
}