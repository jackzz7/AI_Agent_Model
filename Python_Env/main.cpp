#include <windows.h>
#include <stdio.h>

#include"python_API.h"
#include<string.h>

#include <strsafe.h>

#define BUFSIZE 40960

CHAR chBuf[BUFSIZE];
DWORD dwRead, dwWritten;
HANDLE hStdin, hStdout;
BOOL bSuccess;

void ErrorExit(const wchar_t* lpszFunction)

// Format a readable error message, display a message box, 
// and exit from the application.
{
	LPVOID lpMsgBuf;
	LPVOID lpDisplayBuf;
	DWORD dw = GetLastError();

	FormatMessage(
		FORMAT_MESSAGE_ALLOCATE_BUFFER |
		FORMAT_MESSAGE_FROM_SYSTEM |
		FORMAT_MESSAGE_IGNORE_INSERTS,
		NULL,
		dw,
		MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
		(LPTSTR)&lpMsgBuf,
		0, NULL);

	lpDisplayBuf = (LPVOID)LocalAlloc(LMEM_ZEROINIT,
		(lstrlen((LPCTSTR)lpMsgBuf) + lstrlen((LPCTSTR)lpszFunction) + 40) * sizeof(TCHAR));
	StringCchPrintf((LPTSTR)lpDisplayBuf,
		LocalSize(lpDisplayBuf) / sizeof(TCHAR),
		TEXT("%s failed with error %d: %s"),
		lpszFunction, dw, lpMsgBuf);
	MessageBox(NULL, (LPCTSTR)lpDisplayBuf, TEXT("Error"), MB_OK);

	LocalFree(lpMsgBuf);
	LocalFree(lpDisplayBuf);
	ExitProcess(1);
}

void pipe_send_Output(const std::string& s) {
	dwRead = s.length();
	// Write to standard output and stop on error.
	bSuccess = WriteFile(hStdout, s.c_str(), dwRead, &dwWritten, NULL);
	assert(dwRead == dwWritten);
	if (!bSuccess)
		ErrorExit(TEXT("send output error"));
}
CHAR* pipe_receive_Input() {
	
	// Read from standard input and stop on error or no data.
	bSuccess = ReadFile(hStdin, chBuf, BUFSIZE, &dwRead, NULL);

	if (!bSuccess || dwRead == 0)
		ErrorExit(TEXT("receive input error"));
	assert(0 <= dwRead && dwRead < BUFSIZE);
	chBuf[dwRead] = '\0';
	return chBuf;
}

void run_Script() {

	//PyImport_AppendInittab("emb", emb::PyInit_emb);
	//_putenv_s("PYTHONPATH", "C:\\Users\\zz7\\source\\repos\\riichi_python\\Project");
	////_putenv_s("PYTHONPATH", "C:\\Users\\ZERGZ\\source\\repos\\Riichi_API\\Project\\");
	//////Init Python thread support
	//Py_Initialize();
	////Py_Initialize();
	//PyImport_ImportModule("emb");
	//char filename[] = "C:\\Users\\zz7\\source\\repos\\riichi_python\\Project\\battle.py";
	//char filename[] = "C:\\Users\\ZERGZ\\source\\repos\\Riichi_API\\Project\\battle.py";
	std::string str = "";
	// here comes the ***magic***
	{
		// switch sys.stdout to custom handler
		emb::stdout_write_type write = [&](const std::string& s) { pipe_send_Output("@" + s); };
		emb::stdin_read_type read = [&](std::string& s) {
			s = pipe_receive_Input();
			//std::cin >> s;
		};

		std::string script_path = pipe_receive_Input();
		std::string project_path=script_path.substr(0, script_path.find_last_of("\\"));
		pipe_send_Output("@Path Ok");

		PyImport_AppendInittab("emb", emb::PyInit_emb);
		_putenv_s("PYTHONPATH", project_path.c_str());
		Py_Initialize();
		PyImport_ImportModule("emb");
		emb::set_stdout_and_stdin(write, read);

		FILE* fp = _Py_fopen(script_path.c_str(), "r");
		PyRun_SimpleFile(fp, script_path.c_str());
		emb::reset_stdout_and_stdin();
		Py_Finalize();
	}
	//End script
	pipe_send_Output("@close pipe");
	//End Process
	if (strcmp(pipe_receive_Input(), "End Process") != 0)
		ErrorExit(TEXT("End Process error"));
}

int main(void)
{

    hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
    hStdin = GetStdHandle(STD_INPUT_HANDLE);
    if (
        (hStdout == INVALID_HANDLE_VALUE) ||
        (hStdin == INVALID_HANDLE_VALUE)
        )
        ExitProcess(1);

	run_Script();


    // Send something to this process's stdout using printf.
    //printf("\n ** This is a message from the child process. ** \n");

    // This simple algorithm uses the existence of the pipes to control execution.
    // It relies on the pipe buffers to ensure that no data is lost.
    // Larger applications would use more advanced process control.

    //for (;;)
    //{
    //    // Read from standard input and stop on error or no data.
    //    bSuccess = ReadFile(hStdin, chBuf, BUFSIZE, &dwRead, NULL);

    //    if (!bSuccess || dwRead == 0)
    //        break;

    //    // Write to standard output and stop on error.
    //    bSuccess = WriteFile(hStdout, "ok", dwRead, &dwWritten, NULL);

    //    if (!bSuccess)
    //        break;
    //}
    return 0;
}