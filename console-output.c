 #include <windows.h>
 #include <stdio.h>
 
 /*
  * Type of function used. Тип используемой функции
  * (комбинация типов параметры и возвращаемого значения)
  */
  
 typedef void (*importFunction)(int*, double*);
 
 int main(int argc, char **argv)
 {
     importFunction StatNeuroResults;

 
     /* Loading DLL into memory. Загружаем DLL в память */
     HINSTANCE hinstLib = LoadLibrary("men.dll");
     if (hinstLib == NULL) {
         printf("ERROR: unable to load DLL\n");
         return 1;
     }
 
     /* Get a pointer to a function. Получаем указатель на функцию */
     StatNeuroResults = (importFunction)GetProcAddress(hinstLib, "StatNeuroResults");
     if (StatNeuroResults == NULL) {
         printf("ERROR: unable to find DLL function\n");
         return 1;
     }
 
  int index;
  int keyin=1;
  double max;
  
  while(1)
  {
	
	StatNeuroResults(&index,&max);
	
	printf("\n%s","Predicted category = ");
    switch(index)
    {
        case 1: printf("%s\n","2"); break;
        case 2: printf("%s\n","3"); break;
        case 3: printf("%s\n","4"); break;
        case 4: printf("%s\n","5"); break;
        default: break;
    }
    printf("\n%s%.14f","Confidence level = ",max);
	printf("\n\n%s\n","Press any key to make another prediction or enter 0 to quit the program.");
	keyin=getch();
	if(keyin==48)break;
  }

     /*
      * Unload the DLL. Выгружаем DLL (в принципе, это будет сделано
      * автоматически при выходе из программы)
      */
     FreeLibrary(hinstLib);
 
     /* Display the result. Отображаем результат */
   //  printf("The result was: %f\n", result);
 
     return 0;
 }
