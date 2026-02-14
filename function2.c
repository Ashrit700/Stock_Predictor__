#include<stdio.h>
int Sum(int a,int b);
int Sub(int a,int b);
int Div(int a,int b);
int Mul(int a,int b);
int main(){
    int a,b;
    scanf("%d %d",&a,&b);
    int sum,sub,mul,div;
   sum=Sum(a,b);
   sub=Sub(a,b);
   mul=Mul(a,b);
   div=Div(a,b);
    printf("%d\n",sum);
    printf("%d\n",sub);
    printf("%d\n",mul);
    printf("%d\n",div);
    return 0;

}
int Sum(int a,int b){
    int sum;
    sum=a+b;
    return sum;
}
int Sub(int a,int b){
    int sub;
    sub=a-b;
    return sub;
}
int Mul(int a,int b){
    int mul;
    mul=a*b;
    return mul;
}
int Div(int a,int b){
    int div;
    div=a/b;
    return div;
}