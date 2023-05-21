//#include <zip.h>
//#include<stdio.h>
//
//const char* ReadFileFromZip(const char* zipfile, const char* filename_in_zip)
//{
//    //Open the ZIP archive
//    int err = 0;
//    zip* z = zip_open(zipfile, 0, &err);
//    if (err != 0) {
//        printf("error:zip failed,error Code %d\n", err);
//        return 0;
//    }
//
//    //Search for the file of given name
//    const char* name = filename_in_zip;
//    struct zip_stat st;
//    zip_stat_init(&st);
//    zip_stat(z, name, 0, &st);
//
//    //Alloc memory for its uncompressed contents
//    char* contents = new char[st.size];
//
//    //Read the compressed file
//    zip_file* f = zip_fopen(z, name, 0);
//    zip_fread(f, contents, st.size);
//    zip_fclose(f);
//
//    //And close the archive
//    zip_close(z);
//
//    if (st.size == 0)return 0;
//    //printf("%lu %s\n", st.size, contents);
//
//    //Do something with the contents
//    //delete allocated memory
//    //delete[] contents;
//    return contents;
//}