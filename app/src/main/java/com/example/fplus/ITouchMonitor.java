/*
 * This file is auto-generated.  DO NOT MODIFY.
 * Using: C:\\Users\\57615\\AppData\\Local\\Android\\Sdk\\build-tools\\37.0.0\\aidl.exe -I c:\\Users\\57615\\AndroidStudioProjects\\FPlus\\app\\src\\main\\aidl -o c:\\Users\\57615\\AndroidStudioProjects\\FPlus\\poc\\aidl_out4 c:\\Users\\57615\\AndroidStudioProjects\\FPlus\\app\\src\\main\\aidl\\com\\example\\fplus\\ITouchMonitor.aidl
 *
 * DO NOT CHECK THIS FILE INTO A CODE TREE (e.g. git, etc..).
 * ALWAYS GENERATE THIS FILE FROM UPDATED AIDL COMPILER
 * AS A BUILD INTERMEDIATE ONLY. THIS IS NOT SOURCE CODE.
 */
package com.example.fplus;
/**
 * 触摸屏监听服务接口（运行在 Shizuku UserService 进程，shell UID）。
 * 只读打开真实触摸屏，用于判断用户是否正用手触摸屏幕，供"检测让路"使用。
 * 
 * 注意：destroy() 的 transaction code 必须为 16777114（Shizuku 约定），
 * Shizuku 服务停止时会用固定 transaction code 调用它做清理。
 */
public interface ITouchMonitor extends android.os.IInterface
{
  /** Default implementation for ITouchMonitor. */
  public static class Default implements com.example.fplus.ITouchMonitor
  {
    /** Shizuku 保留的销毁方法，transaction code 固定 16777114 */
    @Override public void destroy() throws android.os.RemoteException
    {
    }
    /** 开始监听真实触摸屏 */
    @Override public void start() throws android.os.RemoteException
    {
    }
    /**
     * 一次性获取全部状态：[是否触摸, 归一化X, 归一化Y]。
     * 客户端用后台线程轮询本方法并缓存，避免每帧多次跨进程调用阻塞推理线程。
     */
    @Override public float[] getState() throws android.os.RemoteException
    {
      return null;
    }
    @Override
    public android.os.IBinder asBinder() {
      return null;
    }
  }
  /** Local-side IPC implementation stub class. */
  public static abstract class Stub extends android.os.Binder implements com.example.fplus.ITouchMonitor
  {
    /** Construct the stub and attach it to the interface. */
    @SuppressWarnings("this-escape")
    public Stub()
    {
      this.attachInterface(this, DESCRIPTOR);
    }
    /**
     * Cast an IBinder object into an com.example.fplus.ITouchMonitor interface,
     * generating a proxy if needed.
     */
    public static com.example.fplus.ITouchMonitor asInterface(android.os.IBinder obj)
    {
      if ((obj==null)) {
        return null;
      }
      android.os.IInterface iin = obj.queryLocalInterface(DESCRIPTOR);
      if (((iin!=null)&&(iin instanceof com.example.fplus.ITouchMonitor))) {
        return ((com.example.fplus.ITouchMonitor)iin);
      }
      return new com.example.fplus.ITouchMonitor.Stub.Proxy(obj);
    }
    @Override public android.os.IBinder asBinder()
    {
      return this;
    }
    @Override public boolean onTransact(int code, android.os.Parcel data, android.os.Parcel reply, int flags) throws android.os.RemoteException
    {
      if (code >= android.os.IBinder.FIRST_CALL_TRANSACTION && code <= android.os.IBinder.LAST_CALL_TRANSACTION) {
        data.enforceInterface(DESCRIPTOR);
      }
      switch (code)
      {
        case TRANSACTION_destroy:
        {
          this.destroy();
          reply.writeNoException();
          break;
        }
        case TRANSACTION_start:
        {
          this.start();
          reply.writeNoException();
          break;
        }
        case TRANSACTION_getState:
        {
          float[] _result = this.getState();
          reply.writeNoException();
          reply.writeFloatArray(_result);
          break;
        }
        default:
        {
          return super.onTransact(code, data, reply, flags);
        }
      }
      return true;
    }
    private static final class Proxy implements com.example.fplus.ITouchMonitor
    {
      private android.os.IBinder mRemote;
      Proxy(android.os.IBinder remote)
      {
        mRemote = remote;
      }
      @Override public android.os.IBinder asBinder()
      {
        return mRemote;
      }
      public final java.lang.String getInterfaceDescriptor()
      {
        return DESCRIPTOR;
      }
      /** Shizuku 保留的销毁方法，transaction code 固定 16777114 */
      @Override public void destroy() throws android.os.RemoteException
      {
        android.os.Parcel _data = android.os.Parcel.obtain();
        android.os.Parcel _reply = android.os.Parcel.obtain();
        try {
          _data.writeInterfaceToken(DESCRIPTOR);
          boolean _status = mRemote.transact(Stub.TRANSACTION_destroy, _data, _reply, 0);
          _reply.readException();
        }
        finally {
          _reply.recycle();
          _data.recycle();
        }
      }
      /** 开始监听真实触摸屏 */
      @Override public void start() throws android.os.RemoteException
      {
        android.os.Parcel _data = android.os.Parcel.obtain();
        android.os.Parcel _reply = android.os.Parcel.obtain();
        try {
          _data.writeInterfaceToken(DESCRIPTOR);
          boolean _status = mRemote.transact(Stub.TRANSACTION_start, _data, _reply, 0);
          _reply.readException();
        }
        finally {
          _reply.recycle();
          _data.recycle();
        }
      }
      /**
       * 一次性获取全部状态：[是否触摸, 归一化X, 归一化Y]。
       * 客户端用后台线程轮询本方法并缓存，避免每帧多次跨进程调用阻塞推理线程。
       */
      @Override public float[] getState() throws android.os.RemoteException
      {
        android.os.Parcel _data = android.os.Parcel.obtain();
        android.os.Parcel _reply = android.os.Parcel.obtain();
        float[] _result;
        try {
          _data.writeInterfaceToken(DESCRIPTOR);
          boolean _status = mRemote.transact(Stub.TRANSACTION_getState, _data, _reply, 0);
          _reply.readException();
          _result = _reply.createFloatArray();
        }
        finally {
          _reply.recycle();
          _data.recycle();
        }
        return _result;
      }
    }
    static final int TRANSACTION_destroy = (android.os.IBinder.FIRST_CALL_TRANSACTION + 16777114);
    static final int TRANSACTION_start = (android.os.IBinder.FIRST_CALL_TRANSACTION + 1);
    static final int TRANSACTION_getState = (android.os.IBinder.FIRST_CALL_TRANSACTION + 2);
  }
  /** @hide */
  public static final java.lang.String DESCRIPTOR = "com.example.fplus.ITouchMonitor";
  /** Shizuku 保留的销毁方法，transaction code 固定 16777114 */
  public void destroy() throws android.os.RemoteException;
  /** 开始监听真实触摸屏 */
  public void start() throws android.os.RemoteException;
  /**
   * 一次性获取全部状态：[是否触摸, 归一化X, 归一化Y]。
   * 客户端用后台线程轮询本方法并缓存，避免每帧多次跨进程调用阻塞推理线程。
   */
  public float[] getState() throws android.os.RemoteException;
}
